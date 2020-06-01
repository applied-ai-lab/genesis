# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2019 University of Oxford. All rights reserved.
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
from torch.distributions.normal import Normal

from forge import flags

import modules.blocks as B
import modules.seq_att as seq_att
import modules.decoders as decoders
from modules.component_vae import ComponentVAE

from third_party.sylvester.VAE import VAE

import utils.misc as misc


# Model type
flags.DEFINE_boolean('two_stage', True, 'Use two stages if two, else only one.')
# Priors
flags.DEFINE_boolean('autoreg_prior', True, 'Autoregressive prior.')
flags.DEFINE_boolean('comp_prior', True, 'Component prior.')
# Attention VAE
flags.DEFINE_integer('attention_latents', 64, 'Latent dimension.')
flags.DEFINE_string('enc_norm', 'bn', '{bn, in} - norm type in encoder.')
flags.DEFINE_string('dec_norm', 'bn', '{bn, in} - norm type in decoder.')
# Component VAE
flags.DEFINE_integer('comp_enc_channels', 32, 'Starting number of channels.')
flags.DEFINE_integer('comp_ldim', 16, 'Latent dimension of the VAE.')
flags.DEFINE_integer('comp_dec_channels', 32, 'Num channels in Broadcast Decoder.')
flags.DEFINE_integer('comp_dec_layers', 4, 'Num layers in Broadcast Decoder.')
# Losses
flags.DEFINE_boolean('pixel_bound', True, 'Bound pixel values to [0, 1].')
flags.DEFINE_float('pixel_std1', 0.7, 'StdDev of reconstructed pixels.')
flags.DEFINE_float('pixel_std2', 0.7, 'StdDev of reconstructed pixels.')
flags.DEFINE_boolean('montecarlo_kl', True, 'Evaluate KL via MC samples.')


def load(cfg):
    return Genesis(cfg)


class Genesis(nn.Module):

    def __init__(self, cfg):
        super(Genesis, self).__init__()
        # --- Configuration ---
        # Data dependent config
        self.K_steps = cfg.K_steps
        self.img_size = cfg.img_size
        # Model config
        self.two_stage = cfg.two_stage
        self.autoreg_prior = cfg.autoreg_prior
        self.comp_prior = cfg.comp_prior if self.two_stage else False
        self.ldim = cfg.attention_latents
        self.pixel_bound = cfg.pixel_bound
        # Sanity checks
        self.debug = cfg.debug
        assert cfg.montecarlo_kl == True  # ALWAYS use MC for estimating KL

        # --- Modules ---
        if hasattr(cfg, 'input_channels'):
            input_channels = cfg.input_channels
        else:
            input_channels = 3
        # - Attention core
        att_nin = input_channels
        att_nout = 1
        att_core = VAE(self.ldim, [att_nin, cfg.img_size, cfg.img_size],
                       att_nout, cfg.enc_norm, cfg.dec_norm)
        # - Attention process
        self.att_steps = self.K_steps
        self.att_process = seq_att.LatentSBP(att_core)
        # - Component VAE
        if self.two_stage:
            self.comp_vae = ComponentVAE(
                nout=input_channels, cfg=cfg, act=nn.ELU())
        else:
            self.decoder = decoders.BroadcastDecoder(
                in_chnls=self.ldim, out_chnls=input_channels,
                h_chnls=cfg.comp_dec_channels, num_layers=cfg.comp_dec_layers,
                img_dim=self.img_size, act=nn.ELU())

        # --- Priors ---
        # Optional: Autoregressive prior
        if self.autoreg_prior:
            self.prior_lstm = nn.LSTM(self.ldim, 256)
            self.prior_linear = nn.Linear(256, 2*self.ldim)
        # Optional: Component prior - only relevant for two stage model
        if self.comp_prior and self.two_stage:
            self.prior_mlp = nn.Sequential(
                nn.Linear(self.ldim, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, 2*cfg.comp_ldim))

        # --- Output pixel distribution ---
        std = cfg.pixel_std2 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = cfg.pixel_std1  # first step
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Performs a forward pass in the model.

        Args:
          x (torch.Tensor): input images [batch size, 3, dim, dim]

        Returns:
          recon: reconstructed images [N, 3, H, W]
          losses: 
          stats: 
          att_stats: 
          comp_stats: 
        """

        # --- Predict segmentation masks ---
        log_m_k, log_s_k, att_stats = self.att_process(x, self.att_steps)
        # UGLY: Need to correct the last mask for one stage model wo/ softmax
        # OR when running the two stage model for an additional step
        if len(log_m_k) == self.K_steps+1:
            del log_m_k[-1]
            log_m_k[self.K_steps-1] = log_s_k[self.K_steps-1]
        if self.debug or not self.training:
            assert len(log_m_k) == self.K_steps

        # --- Reconstruct components ---
        if self.two_stage:
            x_r_k, comp_stats = self.comp_vae(x, log_m_k)
        else:
            # x_r_k = [x[:, 1:, :, :] for x in att_stats.x_k]
            z_batched = torch.cat(att_stats.z_k, dim=0)
            x_r_batched = self.decoder(z_batched)
            x_r_k = torch.chunk(x_r_batched, self.K_steps, dim=0)
            if self.pixel_bound:
                x_r_k = [torch.sigmoid(x) for x in x_r_k]
            comp_stats = None

        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        recon = (m_stack * x_r_stack).sum(dim=4)

        # --- Loss terms ---
        losses = AttrDict()
        # -- Reconstruction loss
        losses['err'] = self.x_loss(x, log_m_k, x_r_k, self.std)
        # -- Attention mask KL
        # Using normalising flow, arbitrary posterior
        if 'zm_0_k' in att_stats and 'zm_k_k' in att_stats:
            q_zm_0_k = [Normal(m, s) for m, s in
                        zip(att_stats.mu_k, att_stats.sigma_k)]
            zm_0_k = att_stats.z_0_k
            zm_k_k = att_stats.z_k_k  #TODO(martin) variable name not ideal
            ldj_k = att_stats.ldj_k
        # No flow, Gaussian posterior
        else:
            q_zm_0_k = [Normal(m, s) for m, s in
                        zip(att_stats.mu_k, att_stats.sigma_k)]
            zm_0_k = att_stats.z_k
            zm_k_k = att_stats.z_k
            ldj_k = None
        # Compute loss
        losses['kl_m_k'], p_zm_k = self.mask_latent_loss(
            q_zm_0_k, zm_0_k, zm_k_k, ldj_k, self.prior_lstm, self.prior_linear,
            debug=self.debug or not self.training)
        att_stats['pmu_k'] = [p_zm.mean for p_zm in p_zm_k]
        att_stats['psigma_k'] = [p_zm.scale for p_zm in p_zm_k]
        # Sanity checks
        if self.debug or not self.training:
            assert len(zm_k_k) == self.K_steps
            assert len(zm_0_k) == self.K_steps
            if ldj_k is not None:
                assert len(ldj_k) == self.K_steps
        # -- Component KL
        if self.two_stage:
            if self.comp_prior:
                losses['kl_l_k'] = []
                comp_stats['pmu_k'], comp_stats['psigma_k'] = [], []
                for step, zl in enumerate(comp_stats.z_k):
                    mlp_out = self.prior_mlp(zm_k_k[step])
                    mlp_out = torch.chunk(mlp_out, 2, dim=1)
                    mu = torch.tanh(mlp_out[0])
                    sigma = B.to_prior_sigma(mlp_out[1])
                    p_zl = Normal(mu, sigma)
                    comp_stats['pmu_k'].append(mu)
                    comp_stats['psigma_k'].append(sigma)
                    q_zl = Normal(comp_stats.mu_k[step], comp_stats.sigma_k[step])
                    kld = (q_zl.log_prob(zl) - p_zl.log_prob(zl)).sum(dim=1)
                    losses['kl_l_k'].append(kld)
                # Sanity checks
                if self.debug or not self.training:
                    assert len(comp_stats.z_k) == self.K_steps
                    assert len(comp_stats['pmu_k']) == self.K_steps
                    assert len(comp_stats['psigma_k']) == self.K_steps
                    assert len(losses['kl_l_k']) == self.K_steps
            else:
                raise NotImplementedError

        # Tracking
        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            mx_r_k=[x*logm.exp() for x, logm in zip(x_r_k, log_m_k)])

        # Sanity check that masks sum to one if in debug mode
        if self.debug or not self.training:
            assert len(log_m_k) == self.K_steps
            misc.check_log_masks(log_m_k)

        return recon, losses, stats, att_stats, comp_stats

    @staticmethod
    def x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
        # 1.) Sum over steps for per pixel & channel (ppc) losses
        p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        # TODO(martin): use LogSumExp trick for numerical stability
        err_ppc = -torch.log(log_mx.exp().sum(dim=4))
        # 2.) Sum accross channels and spatial dimensions
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))

    @staticmethod
    def mask_latent_loss(q_zm_0_k, zm_0_k, zm_k_k, ldj_k,
                         prior_lstm=None, prior_linear=None, debug=False):
        num_steps = len(zm_k_k)
        batch_size = zm_k_k[0].size(0)
        latent_dim = zm_k_k[0].size(1)

        # -- Determine prior --
        if prior_lstm is not None and prior_linear is not None:
            # zm_seq shape: (att_steps-2, batch_size, ldim)
            # Do not need the last element in z_k
            zm_seq = torch.cat(
                [zm.view(1, batch_size, -1) for zm in zm_k_k[:-1]], dim=0)
            # lstm_out shape: (att_steps-2, batch_size, state_size)
            # Note: recurrent state is handled internally by LSTM
            lstm_out, _ = prior_lstm(zm_seq)
            # linear_out shape: (att_steps-2, batch_size, 2*ldim)
            linear_out = prior_linear(lstm_out)
            linear_out = torch.chunk(linear_out, 2, dim=2)
            mu_raw = torch.tanh(linear_out[0])
            sigma_raw = B.to_prior_sigma(linear_out[1])
            # Split into K steps, shape: (att_steps-2)*[1, batch_size, ldim]
            mu_k = torch.split(mu_raw, 1, dim=0)
            sigma_k = torch.split(sigma_raw, 1, dim=0)
            # Use standard Normal as prior for first step
            p_zm_k = [Normal(0, 1)]
            # Autoregressive prior for later steps
            for mean, std in zip(mu_k, sigma_k):
                # Remember to remove unit dimension at dim=0
                p_zm_k += [Normal(mean.view(batch_size, latent_dim),
                                  std.view(batch_size, latent_dim))]
            # Sanity checks
            if debug:
                assert zm_seq.size(0) == num_steps-1
        else:
            p_zm_k = num_steps*[Normal(0, 1)]

        # -- Compute KL using Monte Carlo samples for every step k --
        kl_m_k = []
        for step, p_zm in enumerate(p_zm_k):
            log_q = q_zm_0_k[step].log_prob(zm_0_k[step]).sum(dim=1)
            log_p = p_zm.log_prob(zm_k_k[step]).sum(dim=1)
            kld = log_q - log_p
            if ldj_k is not None:
                ldj = ldj_k[step].sum(dim=1)
                kld = kld - ldj
            kl_m_k.append(kld)

        # -- Sanity check --
        if debug:
            assert len(p_zm_k) == num_steps
            assert len(kl_m_k) == num_steps

        return kl_m_k, p_zm_k

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        # --- Mask ---
        # Sample latents
        if self.autoreg_prior:
            zm_k = [Normal(0, 1).sample([batch_size, self.ldim])]
            state = None
            for k in range(1, self.att_steps):
                # TODO(martin) reuse code from forward method?
                lstm_out, state = self.prior_lstm(
                    zm_k[-1].view(1, batch_size, -1), state)
                linear_out = self.prior_linear(lstm_out)
                mu = linear_out[0, :, :self.ldim]
                sigma = B.to_prior_sigma(linear_out[0, :, self.ldim:])
                p_zm = Normal(mu.view([batch_size, self.ldim]),
                              sigma.view([batch_size, self.ldim]))
                zm_k.append(p_zm.sample())
        else:
            p_zm = Normal(0, 1)
            zm_k = [p_zm.sample([batch_size, self.ldim])
                    for _ in range(self.att_steps)]
        # Decode latents into masks
        log_m_k, log_s_k, out_k = self.att_process.masks_from_zm_k(
            zm_k, self.img_size)
        # UGLY: Need to correct the last mask for one stage model wo/ softmax
        # OR when running the two stage model for an additional step
        if len(log_m_k) == self.K_steps+1:
            del log_m_k[-1]
            log_m_k[self.K_steps-1] = log_s_k[self.K_steps-1]
        # Sanity checks. This function is not called at every training step so
        # assert statement do not cause a big slow down in total training time
        assert len(zm_k) == self.K_steps
        assert len(log_m_k) == self.K_steps
        if self.two_stage:
            assert out_k[0].size(1) == 0
        else:
            # assert out_k[0].size(1) == 3
            assert out_k[0].size(1) == 0
        misc.check_log_masks(log_m_k)

        # --- Component appearance ---
        if self.two_stage:
            # Sample latents
            if self.comp_prior:
                zc_k = []
                for zm in zm_k:
                    mlp_out = torch.chunk(self.prior_mlp(zm), 2, dim=1)
                    mu = torch.tanh(mlp_out[0])
                    sigma = B.to_prior_sigma(mlp_out[1])
                    zc_k.append(Normal(mu, sigma).sample())
                # if not self.softmax_attention:
                #     zc_k.append(Normal(0, 1).sample(
                #         [batch_size, self.comp_vae.ldim]))
            else:
                zc_k = [Normal(0, 1).sample([batch_size, self.comp_vae.ldim])
                        for _ in range(K_steps)]
            #  Decode latents into components
            zc_batch = torch.cat(zc_k, dim=0)
            x_batch = self.comp_vae.decode(zc_batch)
            x_k = list(torch.chunk(x_batch, self.K_steps, dim=0))
        else:
            # x_k = out_k
            zm_batched = torch.cat(zm_k, dim=0)
            x_batched = self.decoder(zm_batched)
            x_k = torch.chunk(x_batched, self.K_steps, dim=0)
            if self.pixel_bound:
                x_k = [torch.sigmoid(x) for x in x_k]
        # Sanity check
        assert len(x_k) == self.K_steps
        assert len(log_m_k) == self.K_steps
        if self.two_stage:
            assert len(zc_k) == self.K_steps

        # --- Reconstruct image ---
        x_stack = torch.stack(x_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        generated_image = (m_stack * x_stack).sum(dim=4)

        # Stats
        stats = AttrDict(x_k=x_k, log_m_k=log_m_k, log_s_k=log_s_k,
                         mx_k=[x*m.exp() for x, m in zip(x_k, log_m_k)])
        return generated_image, stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, att_stats, comp_stats = self.forward(image_batch)
        if self.two_stage:
            zm_k = att_stats['z_k'][:self.K_steps-1]
            zc_k = comp_stats['z_k']
            return torch.cat([*zm_k, *zc_k], dim=1)
        else:
            zm_k = att_stats['z_k']
            return torch.cat(zm_k, dim=1)
