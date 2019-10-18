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

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.blocks as B


class BroadcastDecoder(nn.Module):

    def __init__(self, in_chnls, out_chnls, h_chnls, num_layers, img_dim, act):
        super(BroadcastDecoder, self).__init__()
        broad_dim = img_dim + 2*num_layers
        mods = [B.BroadcastLayer(broad_dim),
                nn.Conv2d(in_chnls+2, h_chnls, 3),
                act]
        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(h_chnls, h_chnls, 3), act])
        mods.append(nn.Conv2d(h_chnls, out_chnls, 1))
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        return self.seq(x)
