# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
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
