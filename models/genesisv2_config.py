# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from forge import flags

from modules.unet import UNet
import modules.attention as attention
import modules.blocks as B

from models.genesis_config import Genesis
from models.monet_config import MONet

import utils.misc as misc


# Architecture
flags.DEFINE_integer('feat_dim', 64, 'Number of features and latents.')
# Segmentation
flags.DEFINE_string('kernel', 'gaussian', '{laplacian, gaussian, epanechnikov')
flags.DEFINE_boolean('semiconv', True, 'Use semi-convolutional embeddings.')
flags.DEFINE_boolean('dynamic_K', False, 'Dynamic K.')
# Auxiliary mask consistency loss
flags.DEFINE_boolean('klm_loss', False, 'KL mask regulariser.')
flags.DEFINE_boolean('detach_mr_in_klm', True, 'Detach reconstructed masks.')


def load(cfg):
    return GenesisV2(cfg)


class GenesisV2(nn.Module):

    def __init__(self, cfg):
        super(GenesisV2, self).__init__()
        # Configuration
        self.K_steps = cfg.K_steps
        self.pixel_bound = cfg.pixel_bound
        self.feat_dim = cfg.feat_dim
        self.klm_loss = cfg.klm_loss
        self.detach_mr_in_klm = cfg.detach_mr_in_klm
        self.dynamic_K = cfg.dynamic_K
        self.debug = cfg.debug
        self.multi_gpu = cfg.multi_gpu
        # Encoder
        self.encoder = UNet(
            num_blocks=int(np.log2(cfg.img_size)-1),
            img_size=cfg.img_size,
            filter_start=min(cfg.feat_dim, 64),
            in_chnls=3,
            out_chnls=cfg.feat_dim,
            norm='gn')
        self.encoder.final_conv = nn.Identity()
        self.att_process = attention.InstanceColouringSBP(
            img_size=cfg.img_size,
            kernel=cfg.kernel,
            colour_dim=8,
            K_steps=self.K_steps,
            feat_dim=cfg.feat_dim,
            semiconv=cfg.semiconv)
        self.seg_head = B.ConvGNReLU(cfg.feat_dim, cfg.feat_dim, 3, 1, 1)
        self.feat_head = nn.Sequential(
            B.ConvGNReLU(cfg.feat_dim, cfg.feat_dim, 3, 1, 1),
            nn.Conv2d(cfg.feat_dim, 2*cfg.feat_dim, 1))
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*cfg.feat_dim),
            nn.Linear(2*cfg.feat_dim, 2*cfg.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*cfg.feat_dim, 2*cfg.feat_dim))
        # Decoder
        c = cfg.feat_dim
        self.decoder_module = nn.Sequential(
            B.BroadcastLayer(cfg.img_size // 16),
            nn.ConvTranspose2d(cfg.feat_dim+2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1))
        # --- Prior ---
        self.autoreg_prior = cfg.autoreg_prior
        self.prior_lstm, self.prior_linear = None, None
        if self.autoreg_prior and self.K_steps > 1:
            self.prior_lstm = nn.LSTM(cfg.feat_dim, 4*cfg.feat_dim)
            self.prior_linear = nn.Linear(4*cfg.feat_dim, 2*cfg.feat_dim)
        # --- Output pixel distribution ---
        assert cfg.pixel_std1 == cfg.pixel_std2
        self.std = cfg.pixel_std1

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # --- Extract features ---
        enc_feat, _ = self.encoder(x)
        enc_feat = F.relu(enc_feat)

        # --- Predict attention masks ---
        if self.dynamic_K:
            if batch_size > 1:
                # Iterate over individual elements in batch
                log_m_k = [[] for _ in range(self.K_steps)]
                att_stats, log_s_k = None, None
                for f in torch.split(enc_feat, 1, dim=0):
                    log_m_k_b, _, _ = self.att_process(
                        self.seg_head(f), self.K_steps-1, dynamic_K=True)
                    for step in range(self.K_steps):
                        if step < len(log_m_k_b):
                            log_m_k[step].append(log_m_k_b[step])
                        else:
                            log_m_k[step].append(-1e10*torch.ones([1, 1, H, W]))
                for step in range(self.K_steps):
                    log_m_k[step] = torch.cat(log_m_k[step], dim=0)
                if self.debug:
                    assert len(log_m_k) == self.K_steps
            else:
                log_m_k, log_s_k, att_stats = self.att_process(
                    self.seg_head(enc_feat), self.K_steps-1, dynamic_K=True)
        else:
            log_m_k, log_s_k, att_stats = self.att_process(
                self.seg_head(enc_feat), self.K_steps-1, dynamic_K=False)
            if self.debug:
                assert len(log_m_k) == self.K_steps

        # -- Object features, latents, and KL
        comp_stats = AttrDict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        for log_m in log_m_k:
            mask = log_m.exp()
            # Masked sum
            obj_feat = mask * self.feat_head(enc_feat)
            obj_feat = obj_feat.sum((2, 3))
            # Normalise
            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-5)
            # Posterior
            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            sigma = B.to_sigma(sigma_ps)
            q_z = Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats['mu_k'].append(mu)
            comp_stats['sigma_k'].append(sigma)
            comp_stats['z_k'].append(z)
            comp_stats['q_z_k'].append(q_z)

        # --- Decode latents ---
        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats.z_k)

        # --- Loss terms ---
        losses = AttrDict()
        # -- Reconstruction loss
        losses['err'] = Genesis.x_loss(x, log_m_r_k, x_r_k, self.std)
        mx_r_k = [x*logm.exp() for x, logm in zip(x_r_k, log_m_r_k)]
        # -- Optional: Attention mask loss
        if self.klm_loss:
            if self.detach_mr_in_klm:
                log_m_r_k = [m.detach() for m in log_m_r_k]
            losses['kl_m'] = MONet.kl_m_loss(
                log_m_k=log_m_k, log_m_r_k=log_m_r_k, debug=self.debug)
        # -- Component KL
        losses['kl_l_k'], p_z_k = Genesis.mask_latent_loss(
            comp_stats.q_z_k, comp_stats.z_k,
            prior_lstm=self.prior_lstm, prior_linear=self.prior_linear,
            debug=self.debug)

        # Track quantities of interest
        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k, mx_r_k=mx_r_k,
            instance_seg=torch.argmax(torch.cat(log_m_k, dim=1), dim=1),
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k, dim=1), dim=1))

        # Sanity checks
        if self.debug:
            if not self.dynamic_K:
                assert len(log_m_k) == self.K_steps
                assert len(log_m_r_k) == self.K_steps
            misc.check_log_masks(log_m_k)
            misc.check_log_masks(log_m_r_k)

        if self.multi_gpu:
            # q_z_k is a torch.distribution which doesn't work with the
            # gathering used by DataParallel.
            del comp_stats['q_z_k']

        return recon, losses, stats, att_stats, comp_stats

    def decode_latents(self, z_k):
        # --- Reconstruct components and image ---
        x_r_k, m_r_logits_k = [], []
        for z in z_k:
            dec = self.decoder_module(z)
            x_r_k.append(dec[:, :3, :, :])
            m_r_logits_k.append(dec[:, 3: , :, :])
        # Optional: Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        # --- Reconstruct masks ---
        log_m_r_stack = MONet.get_mask_recon_stack(
            m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps

        # Sample latents
        if self.autoreg_prior:
            z_k = [Normal(0, 1).sample([batch_size, self.feat_dim])]
            state = None
            for k in range(1, K_steps):
                # TODO(martin): reuse code from forward method?
                lstm_out, state = self.prior_lstm(
                    z_k[-1].view(1, batch_size, -1), state)
                linear_out = self.prior_linear(lstm_out)
                linear_out = torch.chunk(linear_out, 2, dim=2)
                linear_out = [item.squeeze(0) for item in linear_out]
                mu = torch.tanh(linear_out[0])
                sigma = B.to_prior_sigma(linear_out[1])
                p_z = Normal(mu.view([batch_size, self.feat_dim]),
                             sigma.view([batch_size, self.feat_dim]))
                z_k.append(p_z.sample())
        else:
            p_z = Normal(0, 1)
            z_k = [p_z.sample([batch_size, self.feat_dim])
                   for _ in range(K_steps)]

        # Decode latents
        recon, x_r_k, log_m_r_k = self.decode_latents(z_k)

        stats = AttrDict(x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return recon, stats
