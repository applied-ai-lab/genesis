################################################################################
# Adapted from https://github.com/riannevdberg/sylvester-flows
#
# Modified by Martin Engelcke
################################################################################

import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride,
                 padding, dilation=1, activation=None,
                 h_norm=None, g_norm=None):
        super(GatedConv2d, self).__init__()
        # Main
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(input_channels, 2*output_channels, kernel_size,
                              stride, padding, dilation)
        # Normalisation
        self.h_norm, self.g_norm = None, None
        # - Hiddens
        if h_norm == 'in':
            self.h_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif h_norm == 'bn':
            self.h_norm = nn.BatchNorm2d(output_channels)
        elif h_norm is None or h_norm == 'none':
            pass
        else:
            raise ValueError("Normalisation option not recognised.")
        # - Gates
        if g_norm == 'in':
            self.g_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif g_norm == 'bn':
            self.g_norm = nn.BatchNorm2d(output_channels)
        elif g_norm is None or g_norm == 'none':
            pass
        else:
            raise ValueError("Normalisation option not recognised.")

    def forward(self, x):
        h, g = torch.chunk(self.conv(x), 2, dim=1)
        # Features
        if self.h_norm is not None:
            h = self.h_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        # Gates
        if self.g_norm is not None:
            g = self.g_norm(g)
        g = self.sigmoid(g)
        # Output
        return h * g


class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride,
                 padding, output_padding=0, dilation=1, activation=None,
                 h_norm=None, g_norm=None):
        super(GatedConvTranspose2d, self).__init__()
        # Main
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.ConvTranspose2d(
            input_channels, 2*output_channels, kernel_size, stride, padding,
            output_padding, dilation=dilation)
        # Normalisation
        # - Hiddens
        self.h_norm, self.g_norm = None, None
        if h_norm == 'in':
            self.h_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif h_norm == 'bn':
            self.h_norm = nn.BatchNorm2d(output_channels)
        elif h_norm is None or h_norm == 'none':
            pass
        else:
            raise ValueError("Normalisation option not recognised.")
        # - Gates
        if g_norm == 'in':
            self.g_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif g_norm == 'bn':
            self.g_norm = nn.BatchNorm2d(output_channels)
        elif g_norm is None or g_norm == 'none':
            pass
        else:
            raise ValueError("Normalisation option not recognised.")

    def forward(self, x):
        h, g = torch.chunk(self.conv(x), 2, dim=1)
        # Features
        if self.h_norm is not None:
            h = self.h_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        # Gates
        if self.g_norm is not None:
            g = self.g_norm(g)
        g = self.sigmoid(g)
        # Output
        return h * g
