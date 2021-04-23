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

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.blocks import INConvBlock, Flatten


class UNet(nn.Module):

    def __init__(self, num_blocks, filter_start=32):
        super(UNet, self).__init__()
        # TODO(martin) generalise to other cases if nedded
        c = filter_start
        if num_blocks == 4:
            self.down = nn.ModuleList([
                INConvBlock(4, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c)
            ])
        elif num_blocks == 5:
            self.down = nn.ModuleList([
                INConvBlock(4, c),
                INConvBlock(c, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c)
            ])
        elif num_blocks == 6:
            self.down = nn.ModuleList([
                INConvBlock(4, c),
                INConvBlock(c, c),
                INConvBlock(c, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c)
            ])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(4*4*2*c, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4*4*2*c), nn.ReLU()
        )
        self.final_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        return self.final_conv(x_up), {}
