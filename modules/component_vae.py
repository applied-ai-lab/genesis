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

import modules.blocks as B
from modules.encoders import MONetCompEncoder
from modules.decoders import BroadcastDecoder


class ComponentVAE(nn.Module):

    def __init__(self, nout, cfg, act):
        super(ComponentVAE, self).__init__()
        self.ldim = cfg.comp_ldim  # paper uses 16
        self.montecarlo = cfg.montecarlo_kl
        self.pixel_bound = cfg.pixel_bound
        # Sub-Modules
        self.encoder_module = MONetCompEncoder(cfg=cfg, act=act)
        self.decoder_module = BroadcastDecoder(
            in_chnls=self.ldim,
            out_chnls=nout,
            h_chnls=cfg.comp_dec_channels,
            num_layers=cfg.comp_dec_layers,
            img_dim=cfg.img_size,
            act=act
        )

    def forward(self, x, log_mask):
        """
        Args:
            x (torch.Tensor): Input to reconstruct [batch size, 3, dim, dim]
            log_mask (torch.Tensor or list of torch.Tensors):
                Mask to reconstruct [batch size, 1, dim, dim]
        """
        # -- Check if inputs are lists
        K = 1
        b_sz = x.size(0)
        if isinstance(log_mask, list) or isinstance(log_mask, tuple):
            K = len(log_mask)
            # Repeat x along batch dimension
            x = x.repeat(K, 1, 1, 1)
            # Concat log_m_k along batch dimension
            log_mask = torch.cat(log_mask, dim=0)

        # -- Encode
        x = torch.cat((log_mask, x), dim=1)  # Concat along feature dimension
        mu, sigma = self.encode(x)

        # -- Sample latents
        q_z = Normal(mu, sigma)
        # z - [batch_size * K, l_dim] with first axis: b0,k0 -> b0,k1 -> ...
        z = q_z.rsample()

        # -- Decode
        # x_r, m_r_logits = self.decode(z)
        x_r = self.decode(z)

        # -- Track quantities of interest and return
        x_r_k = torch.chunk(x_r, K, dim=0)
        z_k = torch.chunk(z, K, dim=0)
        mu_k = torch.chunk(mu, K, dim=0)
        sigma_k = torch.chunk(sigma, K, dim=0)
        stats = AttrDict(mu_k=mu_k, sigma_k=sigma_k, z_k=z_k)
        return x_r_k, stats

    def encode(self, x):
        x = self.encoder_module(x)
        mu, sigma_ps = torch.chunk(x, 2, dim=1)
        sigma = B.to_sigma(sigma_ps)
        return mu, sigma

    def decode(self, z):
        x_hat = self.decoder_module(z)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def sample(self, batch_size=1, steps=1):
        raise NotImplementedError
