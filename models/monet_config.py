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
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np

from forge import flags

from modules.unet import UNet
import modules.attention as attention
from modules.component_vae import ComponentVAE
from models.genesis_config import Genesis
from utils import misc


# Attention network
flags.DEFINE_integer('filter_start', 32, 'Starting number of channels in UNet.')
flags.DEFINE_string('prior_mode', 'softmax', '{scope, softmax}')


def load(cfg):
    return MONet(cfg)


class MONet(nn.Module):

    def __init__(self, cfg):
        super(MONet, self).__init__()
        # Configuration
        self.K_steps = cfg.K_steps
        self.prior_mode = cfg.prior_mode
        self.mckl = cfg.montecarlo_kl
        self.debug = cfg.debug
        self.pixel_bound = cfg.pixel_bound
        # Sub-Modules
        # - Attention Network
        if not hasattr(cfg, 'filter_start'):
            cfg['filter_start'] = 32
        core = UNet(
            num_blocks=int(np.log2(cfg.img_size)-1),
            img_size=cfg.img_size,
            filter_start=cfg.filter_start,
            in_chnls=4,
            out_chnls=1,
            norm='in')
        self.att_process = attention.SimpleSBP(core)
        # - Component VAE
        self.comp_vae = ComponentVAE(nout=4, cfg=cfg, act=nn.ReLU())
        self.comp_vae.pixel_bound = False
        # Initialise pixel output standard deviations
        std = cfg.pixel_std2 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = cfg.pixel_std1  # first step
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images [batch size, 3, dim, dim]
        """
        # --- Predict segmentation masks ---
        log_m_k, log_s_k, att_stats = self.att_process(x, self.K_steps-1)

        # --- Reconstruct components ---
        x_m_r_k, comp_stats = self.comp_vae(x, log_m_k)
        # Split into appearances and mask prior
        x_r_k = [item[:, :3, :, :] for item in x_m_r_k]
        m_r_logits_k = [item[:, 3:, :, :] for item in x_m_r_k]
        # Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]

        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        recon = (m_stack * x_r_stack).sum(dim=4)

        # --- Reconstruct masks ---
        log_m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]

        # --- Loss terms ---
        losses = AttrDict()
        # -- Reconstruction loss
        losses['err'] = Genesis.x_loss(x, log_m_k, x_r_k, self.std)
        # -- Attention mask KL
        losses['kl_m'] = self.kl_m_loss(log_m_k=log_m_k, log_m_r_k=log_m_r_k)
        # -- Component KL
        q_z_k = [Normal(m, s) for m, s in 
                 zip(comp_stats.mu_k, comp_stats.sigma_k)]
        kl_l_k = misc.get_kl(
            comp_stats.z_k, q_z_k, len(q_z_k)*[Normal(0, 1)], self.mckl)
        losses['kl_l_k'] = [kld.sum(1) for kld in kl_l_k]

        # Track quantities of interest
        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k,
            mx_r_k=[x*logm.exp() for x, logm in zip(x_r_k, log_m_k)])

        # Sanity check that masks sum to one if in debug mode
        if self.debug:
            assert len(log_m_k) == self.K_steps
            assert len(log_m_r_k) == self.K_steps
            misc.check_log_masks(log_m_k)
            misc.check_log_masks(log_m_r_k)

        return recon, losses, stats, att_stats, comp_stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    @staticmethod
    def get_mask_recon_stack(m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_s = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == len(m_r_logits_k) - 1:
                    log_m_r_k.append(log_s)
                else:
                    log_a = F.logsigmoid(logits)
                    log_neg_a = F.logsigmoid(-logits)
                    log_m_r_k.append(log_s + log_a)
                    log_s = log_s +  log_neg_a
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")

    @staticmethod
    def kl_m_loss(log_m_k, log_m_r_k, debug=False):
        if debug:
            assert len(log_m_k) == len(log_m_r_k)
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        # Lower bound to 1e-5 to avoid infinities
        m_stack = torch.max(m_stack, torch.tensor(1e-5))
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5))
        q_m = Categorical(m_stack.view(-1, len(log_m_k)))
        p_m = Categorical(m_r_stack.view(-1, len(log_m_k)))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        # Sample latents
        z_batched = Normal(0, 1).sample((batch_size*K_steps, self.comp_vae.ldim))
        # Pass latent through decoder
        x_hat_batched = self.comp_vae.decode(z_batched)
        # Split into appearances and masks
        x_r_batched = x_hat_batched[:, :3, :, :]
        m_r_logids_batched = x_hat_batched[:, 3:, :, :]
        # Apply pixel bound to appearances
        if self.pixel_bound:
            x_r_batched = torch.sigmoid(x_r_batched)
        # Chunk into K steps
        x_r_k = torch.chunk(x_r_batched, K_steps, dim=0)
        m_r_logits_k = torch.chunk(m_r_logids_batched, K_steps, dim=0)
        # Normalise masks
        m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=False)
        # Apply masking and sum to get generated image
        x_r_stack = torch.stack(x_r_k, dim=4)
        gen_image = (m_r_stack * x_r_stack).sum(dim=4)
        # Tracking
        log_m_r_k = [item.squeeze(dim=4) for item in
                     torch.split(m_r_stack.log(), 1, dim=4)]
        stats = AttrDict(gen_image=gen_image, x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return gen_image, stats
