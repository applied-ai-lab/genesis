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
from torch.distributions.normal import Normal

from forge import flags

from modules.blocks import Flatten
from modules.decoders import BroadcastDecoder
from third_party.sylvester.VAE import VAE


# GatedConvVAE
flags.DEFINE_integer('latent_dimension', 64, 'Latent channels.')
flags.DEFINE_boolean('broadcast_decoder', False,
                     'Use broadcast decoder instead of deconv.')
# Losses
flags.DEFINE_boolean('pixel_bound', True, 'Bound pixel values to [0, 1].')
flags.DEFINE_float('pixel_std', 0.7, 'StdDev of reconstructed pixels.')


def load(cfg):
    return BaselineVAE(cfg)


class BaselineVAE(nn.Module):

    def __init__(self, cfg):
        super(BaselineVAE, self).__init__()
        cfg.K_steps = None
        # Configuration
        self.ldim = cfg.latent_dimension
        self.pixel_std = cfg.pixel_std
        self.pixel_bound = cfg.pixel_bound
        self.debug = cfg.debug
        # Module
        nin = cfg.input_channels if hasattr(cfg, 'input_channels') else 3
        self.vae = VAE(self.ldim, [nin, cfg.img_size, cfg.img_size], nin)
        if cfg.broadcast_decoder:
            self.vae.p_x_nn = nn.Sequential(
                Flatten(),
                BroadcastDecoder(in_chnls=self.ldim, out_chnls=64, h_chnls=64,
                                 num_layers=4, img_dim=cfg.img_size,
                                 act=nn.ELU()),
                nn.ELU()
            )
            self.vae.p_x_mean = nn.Conv2d(64, nin, 1, 1, 0)

    def forward(self, x):
        """ x (torch.Tensor): Input images [batch size, 3, dim, dim] """
        # Forward propagation
        recon, stats = self.vae(x)
        if self.pixel_bound:
            recon = torch.sigmoid(recon)
        # Reconstruction loss
        p_xr = Normal(recon, self.pixel_std)
        err = -p_xr.log_prob(x).sum(dim=(1, 2, 3))
        # KL divergence loss
        p_z = Normal(0, 1)
        # TODO(martin): the parsing below is not very intuitive
        # -- No flow
        if 'z' in stats:
            q_z = Normal(stats.mu, stats.sigma)
            kl = q_z.log_prob(stats.z) - p_z.log_prob(stats.z)
            kl = kl.sum(dim=1)
        # -- Using normalising flow
        else:
            q_z_0 = Normal(stats.mu_0, stats.sigma_0)
            kl = q_z_0.log_prob(stats.z_0) - p_z.log_prob(stats.z_k)
            kl = kl.sum(dim=1) - stats.ldj
        # Tracking
        losses = AttrDict(err=err, kl_l=kl)
        return recon, losses, stats, None, None

    def sample(self, batch_size, *args, **kwargs):
        # Sample z
        z = Normal(0, 1).sample([batch_size, self.ldim])
        # Decode z
        x = self.vae.decode(z)
        if self.pixel_bound:
            x = torch.sigmoid(x)
        return x, AttrDict(z=z)

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, stats, _, _ = self.forward(image_batch)
        return stats.z
