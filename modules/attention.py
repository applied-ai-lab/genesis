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

import numpy as np

from modules import blocks as B


class SimpleSBP(nn.Module):

    def __init__(self, core):
        super(SimpleSBP, self).__init__()
        self.core = core

    def forward(self, x, steps_to_run):
        # Initialise lists to store tensors over K steps
        log_m_k = []
        stats_k = []
        # Set initial scope to all ones, so log scope is all zeros
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        # Loop over steps
        for step in range(steps_to_run):
            # Compute mask and update scope. Last step is different
            # Compute a_logits given input and current scope
            core_out, stats = self.core(torch.cat((x, log_s_k[step]), dim=1))
            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[step] + log_a)
            # Update scope given attentikon
            log_s_k.append(log_s_k[step] + log_neg_a)
            # Track stats
            stats_k.append(stats)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Convert list of dicts into dict of lists
        stats = AttrDict()
        for key in stats_k[0]:
            stats[key+'_k'] = [s[key] for s in stats_k]
        return log_m_k, log_s_k, stats

    def masks_from_zm_k(self, zm_k, img_size):
        # zm_k: K*(batch_size, ldim)
        b_sz = zm_k[0].size(0)
        log_m_k = []
        log_s_k = [torch.zeros(b_sz, 1, img_size, img_size)]
        other_k = []
        # TODO(martin): parallelise decoding
        for zm in zm_k:
            core_out = self.core.decode(zm)
            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Take rest of channels for other
            other_k.append(core_out[:, 1:, :, :])
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[-1] + log_a)
            # Update scope given attention
            log_s_k.append(log_s_k[-1] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        return log_m_k, log_s_k, other_k


class LatentSBP(SimpleSBP):

    def __init__(self, core):
        super(LatentSBP, self).__init__(core)
        self.lstm = nn.LSTM(core.z_size+256, 2*core.z_size)
        self.linear = nn.Linear(2*core.z_size, 2*core.z_size)

    def forward(self, x, steps_to_run):
        h = self.core.q_z_nn(x)
        bs = h.size(0)
        h = h.view(bs, -1)
        mean_0 = self.core.q_z_mean(h)
        var_0 = self.core.q_z_var(h)
        z, q_z = self.core.reparameterize(mean_0, var_0)
        z_k = [z]
        q_z_k = [q_z]
        state = None
        for step in range(1, steps_to_run):
            h_and_z = torch.cat([h, z_k[-1]], dim=1)
            lstm_out, state = self.lstm(h_and_z.view(1, bs, -1), state)
            linear_out = self.linear(lstm_out)[0, :, :]
            linear_out = torch.chunk(linear_out, 2, dim=1)
            mean_k = linear_out[0]
            var_k = B.to_var(linear_out[1])
            z, q_z = self.core.reparameterize(mean_k, var_k)
            z_k.append(z)
            q_z_k.append(q_z)
        # Initialise lists to store tensors over K steps
        log_m_k = []
        stats_k = []
        # Set initial scope to all ones, so log scope is all zeros
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        # Run decoder in parallel
        z_batch = torch.cat(z_k, dim=0)
        core_out_batch = self.core.decode(z_batch)
        core_out = torch.chunk(core_out_batch, steps_to_run, dim=0)
        # Compute masks
        for step, (z, q_z, out) in enumerate(zip(z_k, q_z_k, core_out)):
            # Compute a_logits given input and current scope
            stats = AttrDict(x=out, mu=q_z.mean, sigma=q_z.scale, z=z)
            # Take first channel for masks
            a_logits = out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[step] + log_a)
            # Update scope given attention
            log_s_k.append(log_s_k[step] + log_neg_a)
            # Track stats
            stats_k.append(stats)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Convert list of dicts into dict of lists
        stats = AttrDict()
        for key in stats_k[0]:
            stats[key+'_k'] = [s[key] for s in stats_k]
        return log_m_k, log_s_k, stats


class InstanceColouringSBP(nn.Module):

    def __init__(self, img_size, kernel='laplacian',
                 colour_dim=8, K_steps=None, feat_dim=None,
                 semiconv=True):
        super(InstanceColouringSBP, self).__init__()
        # Config
        self.img_size = img_size
        self.kernel = kernel
        self.colour_dim = colour_dim
        # Initialise kernel sigma
        if self.kernel == 'laplacian':
            sigma_init = 1.0 / (np.sqrt(K_steps)*np.log(2))
        elif self.kernel == 'gaussian':
            sigma_init = 1.0 / (K_steps*np.log(2))
        elif self.kernel == 'epanechnikov':
            sigma_init = 2.0 / K_steps
        else:
            return ValueError("No valid kernel.")
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        # Colour head
        if semiconv:
            self.colour_head = B.SemiConv(feat_dim, self.colour_dim, img_size)
        else:
            self.colour_head = nn.Conv2d(feat_dim, self.colour_dim, 1)

    def forward(self, features, steps_to_run, debug=False,
                dynamic_K=False, *args, **kwargs):
        batch_size = features.size(0)
        stats = AttrDict()
        if isinstance(features, tuple):
            features = features[0]
        if dynamic_K:
            assert batch_size == 1
        # Get colours
        colour_out = self.colour_head(features)
        if isinstance(colour_out, tuple):
            colour, delta = colour_out
        else:
            colour, delta = colour_out, None
        # Sample from uniform to select random pixels as seeds
        rand_pixel = torch.empty(batch_size, 1, *colour.shape[2:])
        rand_pixel = rand_pixel.uniform_()
        # Run SBP
        seed_list = []
        log_m_k = []
        log_s_k = [torch.zeros(batch_size, 1, self.img_size, self.img_size)]
        for step in range(steps_to_run):
            # Determine seed
            scope = F.interpolate(log_s_k[step].exp(), size=colour.shape[2:],
                                  mode='bilinear', align_corners=False)
            pixel_probs = rand_pixel * scope
            rand_max = pixel_probs.flatten(2).argmax(2).flatten()
            # TODO(martin): parallelise this
            seed = torch.empty((batch_size, self.colour_dim))
            for bidx in range(batch_size):
                seed[bidx, :] = colour.flatten(2)[bidx, :, rand_max[bidx]]
            seed_list.append(seed)
            # Compute masks
            if self.kernel == 'laplacian':
                distance = B.euclidian_distance(colour, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.kernel == 'gaussian':
                distance = B.squared_distance(colour, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.kernel == 'epanechnikov':
                distance = B.squared_distance(colour, seed)
                alpha = (1 - distance / self.log_sigma.exp()).relu()
            else:
                raise ValueError("No valid kernel.")
            alpha = alpha.unsqueeze(1)
            # Sanity checks
            if debug:
                assert alpha.max() <= 1, alpha.max()
                assert alpha.min() >= 0, alpha.min()
            # Clamp mask values to [0.01, 0.99] for numerical stability
            # TODO(martin): clamp less aggressively?
            alpha = B.clamp_preserve_gradients(alpha, 0.01, 0.99)
            # SBP update
            log_a = torch.log(alpha)
            log_neg_a = torch.log(1 - alpha)
            log_m = log_s_k[step] + log_a
            if dynamic_K and log_m.exp().sum() < 20:
                break
            log_m_k.append(log_m)
            log_s_k.append(log_s_k[step] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Accumulate stats
        stats.update({'colour': colour, 'delta': delta, 'seeds': seed_list})
        return log_m_k, log_s_k, stats
