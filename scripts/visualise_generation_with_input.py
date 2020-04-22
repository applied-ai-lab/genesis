#!/usr/bin/env python3

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

import os
from os import path as osp
import random
from attrdict import AttrDict

import torch
import numpy as np
import matplotlib.pyplot as plt

import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

from utils.plotting import plot


# Data & model config
flags.DEFINE_string('data_config', 'datasets/multid_config.py',
                    'Path to a data config file.')
flags.DEFINE_string('model_config', 'models/genesis_config.py',
                    'Path to a model config file.')
# Trained model
flags.DEFINE_string('model_dir', 'checkpoints/pretrained_multid',
                    'Path to model directory.')
flags.DEFINE_string('model_file', 'multid_model.ckpt', 'Name of model file.')

# Visualize_generation
flags.DEFINE_integer('num_inputs', 10,
                        'Number of inputs to visualize.')
# Don't change this
flags.DEFINE_integer('batch_size', 1, 'Mini-batch size.')


def main():
    # Parse flags
    config = forge.config()
    # Restore flags of pretrained model
    flag_path = osp.join(config.model_dir, 'flags.json')
    fprint(f"Restoring flags from {flag_path}")
    pretrained_flags = AttrDict(fet.json_load(flag_path))
    pretrained_flags.batch_size = 1
    pretrained_flags.gpu = False
    pretrained_flags.debug = True
    fet.print_flags()

    # Fix seeds. Always first thing to be done after parsing the config!
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    _, _, test_loader = fet.load(config.data_config, config)

    # Load model
    model = fet.load(config.model_config, pretrained_flags)
    model_path = osp.join(config.model_dir, config.model_file)
    fprint(f"Restoring model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_1', None)
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_2', None)
    model.load_state_dict(model_state_dict)
    fprint(model)

    # Visualise
    model.eval()

    # note that the batch size should be 1
    for count, batch in enumerate(test_loader):
        if count >= config.num_inputs:
            break

        cur_image_input = batch['input']

        # passing in input
        output, _, stats, _, _ = model(cur_image_input)

        # setting up figure
        fig, axes = plt.subplots(nrows=4, ncols=1+pretrained_flags.K_steps)     

        # input
        plot(axes, 0, 0, cur_image_input, title='Input image', fontsize=12)
        # generated
        plot(axes, 1, 0, output, title='Generated scene', fontsize=12)
        # empty plots
        plot(axes, 2, 0, fontsize=12)
        plot(axes, 3, 0, fontsize=12)
        
        # the components
        x_k = stats['x_r_k']
        log_m_k = stats['log_m_k']
        mx_k = [x*m.exp() for x, m in zip(x_k, log_m_k)]
        log_s_k = stats['log_s_k'] if 'log_s_k' in stats else None

        # Put K generation steps in separate subfigures
        for step in range(pretrained_flags.K_steps):
            mx_step = mx_k[step]
            x_step = x_k[step]
            m_step = log_m_k[step].exp()
            if log_s_k:
                s_step = log_s_k[step].exp()

            pre = 'Mask x RGB ' if step == 0 else ''
            plot(axes, 0, 1+step, mx_step, pre+f'k={step+1}', fontsize=12)
            pre = 'RGB ' if step == 0 else ''
            plot(axes, 1, 1+step, x_step, pre+f'k={step+1}', fontsize=12)
            pre = 'Mask ' if step == 0 else ''
            plot(axes, 2, 1+step, m_step, pre+f'k={step+1}', True, fontsize=12)
            if log_s_k:
                pre = 'Scope ' if step == 0 else ''
                plot(axes, 3, 1+step, s_step, pre+f'k={step+1}', True,
                     axis=step == 0, fontsize=12)

        # Beautify and show figure
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()  # for Qt backend
        plt.show()


if __name__ == "__main__":
    main()
