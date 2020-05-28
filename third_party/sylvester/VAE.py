################################################################################
# Adapted from https://github.com/riannevdberg/sylvester-flows
#
# Modified by Martin Engelcke
################################################################################

from attrdict import AttrDict

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from third_party.sylvester.layers import GatedConv2d, GatedConvTranspose2d

from modules.blocks import ToVar


def build_gc_encoder(cin, cout, stride, cfc, kfc, hn=None, gn=None):
    assert len(cin) == len(cout) and len(cin) == len(stride)
    layers = []
    for l, (i, o, s) in enumerate(zip(cin, cout, stride)):
        layers.append(GatedConv2d(i, o, 5, s, 2, h_norm=hn, g_norm=gn))
    layers.append(GatedConv2d(cout[-1], cfc, kfc, 1, 0))
    return nn.Sequential(*layers)


def build_gc_decoder(cin, cout, stride, zdim, kz, hn=None, gn=None):
    assert len(cin) == len(cout) and len(cin) == len(stride)
    layers = [GatedConvTranspose2d(zdim, cin[0], kz, 1, 0)]
    for l, (i, o, s) in enumerate(zip(cin, cout, stride)):
        layers.append(GatedConvTranspose2d(i, o, 5, s, 2, s-1,
                                           h_norm=hn, g_norm=gn))
    return nn.Sequential(*layers)


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, z_size, input_size, nout,
                 enc_norm=None, dec_norm=None):
        super(VAE, self).__init__()

        # extract model settings from args
        self.z_size = z_size
        self.input_size = input_size
        if nout is not None:
            self.nout = nout
        else:
            self.nout = input_size[0]
        self.enc_norm = enc_norm
        self.dec_norm = dec_norm

        if self.input_size[1] == 32 and self.input_size[2] == 32:
            self.last_kernel_size = 8
            strides = [1, 2, 1, 2, 1]
        elif self.input_size[1] == 64 and self.input_size[2] == 64:
            self.last_kernel_size = 16
            strides = [1, 2, 1, 2, 1]
        elif self.input_size[1] == 128 and self.input_size[2] == 128:
            self.last_kernel_size = 16
            strides = [2, 2, 2, 1, 1]
        elif self.input_size[1] == 256 and self.input_size[2] == 256:
            self.last_kernel_size = 16
            strides = [2, 2, 2, 2, 1]
        else:
            raise ValueError('Invalid input size.')

        self.q_z_nn_output_dim = 256

        # Build encoder
        cin =  [self.input_size[0], 32, 32, 64, 64]
        cout = [32,                 32, 64, 64, 64]
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder(
            cin, cout, strides)

        # Build decoder
        cin =  [64, 64, 32, 32, 32]
        cout = [64, 32, 32, 32, 32]
        self.p_x_nn, self.p_x_mean = self.create_decoder(
            cin, cout, list(reversed(strides)))

        # log-det-jacobian = 0 without flows
        self.log_det_j = torch.tensor(0)

    def create_encoder(self, cin, cout, strides):
        """
        Helper function to create the elemental blocks for the encoder.
        Creates a gated convnet encoder.
        the encoder expects data as input of shape:
        (batch_size, num_channels, width, height).
        """

        q_z_nn = build_gc_encoder(
            cin, cout, strides, self.q_z_nn_output_dim, self.last_kernel_size,
            hn=self.enc_norm, gn=self.enc_norm
        )
        q_z_mean = nn.Linear(256, self.z_size)
        q_z_var = nn.Sequential(
            nn.Linear(256, self.z_size),
            ToVar(),
        )
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self, cin, cout, strides):
        """
        Helper function to create the elemental blocks for the decoder.
        Creates a gated convnet decoder.
        """

        p_x_nn = build_gc_decoder(
            cin, cout, strides, self.z_size, self.last_kernel_size,
            hn=self.dec_norm, gn=self.dec_norm
        )
        p_x_mean = nn.Conv2d(cout[-1], self.nout, 1, 1, 0)
        return p_x_nn, p_x_mean

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """

        q_z = Normal(mu, var.sqrt())
        z = q_z.rsample()
        return z, q_z

    def encode(self, x):
        """
        Encoder expects following data shapes as input:
        shape = (batch_size, num_channels, width, height)
        """

        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z, q_z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        stats = AttrDict(x=x_mean, mu=z_mu, sigma=z_var.sqrt(), z=z)
        return x_mean, stats
