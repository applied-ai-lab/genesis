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

def clamp_preserve_gradients(x, lower, upper):
    # From: http://docs.pyro.ai/en/0.3.3/_modules/pyro/distributions/iaf.html
    return x + (x.clamp(lower, upper) - x).detach()

def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8

def to_var(x):
    return to_sigma(x)**2

def to_prior_sigma(x, simgoid_bias=4.0, eps=1e-4):
    """
    This parameterisation bounds sigma of a learned prior to [eps, 1+eps].
    The default sigmoid_bias of 4.0 initialises sigma to be close to 1.0.
    The default eps prevents instability as sigma -> 0.
    """
    return torch.sigmoid(x + simgoid_bias) + eps

def flatten(x):
    return x.view(x.size(0), -1)

def unflatten(x):
    return x.view(x.size(0), -1, 1, 1)

def pixel_coords(img_size):
    g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, img_size),
                              torch.linspace(-1, 1, img_size))
    g_1 = g_1.view(1, 1, img_size, img_size)
    g_2 = g_2.view(1, 1, img_size, img_size)
    return torch.cat((g_1, g_2), dim=1)

def euclidian_norm(x):
    # Clamp before taking sqrt for numerical stability
    return clamp_preserve_gradients((x**2).sum(1), 1e-10, 1e10).sqrt()

def euclidian_distance(embedA, embedB):
    # Unflatten if needed if one is an image and the other a vector
    # Assumes inputs are batches
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return euclidian_norm(embedA - embedB)

def squared_distance(embedA, embedB):
    # Unflatten if needed if one is an image and the other a vector
    # Assumes inputs are batches
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return ((embedA - embedB)**2).sum(1)

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

class ScalarGate(nn.Module):
    def __init__(self, init=0.0):
        super(ScalarGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(init))
    def forward(self, x):
        return self.gate * x

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
        # TODO(martin): avoid duplication
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

class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )

class SemiConv(nn.Module):
    def __init__(self, nin, nout, img_size):
        super(SemiConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1)
        self.gate = ScalarGate()
        coords = pixel_coords(img_size)
        zeros = torch.zeros(1, nout-2, img_size, img_size)
        self.uv = torch.cat((zeros, coords), dim=1)
    def forward(self, x):
        out = self.gate(self.conv(x))
        delta = out[:, -2:, :, :]
        return out + self.uv.to(out.device), delta
