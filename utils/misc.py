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

import sys
import time
import datetime

import torch
from torch.distributions.kl import kl_divergence

import tensorflow as tf

import numpy as np

from forge.experiment_tools import fprint


def loader_throughput(loader, num_batches=100, burn_in=5):
    assert num_batches > 0
    if burn_in is None:
        burn_in = num_batches // 10
    num_samples = 0
    fprint(f"Train loader throughput stats on {num_batches} batches...")
    for i, batch in enumerate(loader):
        if i == burn_in:
            timer = time.time()
        if i >= burn_in:
            num_samples += batch['input'].size(0)
        if i == num_batches + burn_in:
            break
    dt = time.time() - timer
    spb = dt / num_batches
    ips = num_samples / dt
    fprint(f"{spb:.3f} s/b, {ips:.1f} im/s")


def get_kl(z, q_z, p_z, montecarlo):
    if isinstance(q_z, list) or isinstance(q_z, tuple):
        assert len(q_z) == len(p_z)
        kl = []
        for i in range(len(q_z)):
            if montecarlo:
                assert len(q_z) == len(z)
                kl.append(get_mc_kl(z[i], q_z[i], p_z[i]))
            else:
                kl.append(kl_divergence(q_z[i], p_z[i]))
        return kl
    elif montecarlo:
        return get_mc_kl(z, q_z, p_z)
    return kl_divergence(q_z, p_z)


def get_mc_kl(z, q_z, p_z):
    return q_z.log_prob(z) - p_z.log_prob(z)


def check_log_masks(log_m_k):
    summed_masks = torch.stack(log_m_k, dim=4).exp().sum(dim=4)
    summed_masks = summed_masks.clone().data.cpu().numpy()
    flat = summed_masks.flatten()
    diff = flat - np.ones_like(flat)
    idx = np.argmax(diff)
    max_diff = diff[idx]
    if max_diff > 1e-3 or np.any(np.isnan(flat)):
        print("Max difference: {}".format(max_diff))
        for i, log_m in enumerate(log_m_k):
            mask_k = log_m.exp().data.cpu().numpy()
            print("Mask value at k={}: {}".format(i, mask_k.flatten()[idx]))
        raise ValueError("Masks do not sum to 1.0. Not close enough.")
