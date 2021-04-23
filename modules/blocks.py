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


def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8

def to_prior_sigma(x, simgoid_bias=4.0, eps=1e-4):
    """
    This parameterisation bounds sigma of a learned prior to [eps, 1+eps].
    The default sigmoid_bias of 4.0 initialises sigma to be close to 1.0.
    The default eps prevents instability as sigma -> 0.
    """
    return torch.sigmoid(x + simgoid_bias) + eps

def to_var(x):
    return to_sigma(x)**2

class ToSigma(nn.Module):
    def __init__(self):
        super(ToSigma, self).__init__()
    def forward(self, x):
        return to_sigma(x)

class ToVar(nn.Module):
    def __init__(self):
        super(ToVar, self).__init__()
    def forward(self, x):
        return to_var(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)

class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)
    def forward(self, x):
        b_sz = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return self.pixel_coords(x)

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))
        self.g_1 = g_1.view((1, 1) + g_1.shape)
        self.g_2 = g_2.view((1, 1) + g_2.shape)
    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, g_1, g_2), dim=1)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)

class INConvBlock(nn.Module):
    def __init__(self, nin, nout, stride=1, instance_norm=True, act=nn.ReLU()):
        super(INConvBlock, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)
