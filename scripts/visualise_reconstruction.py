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
flags.DEFINE_string('data_config', 'datasets/gqn_config.py',
                    'Path to a data config file.')
flags.DEFINE_string('model_config', 'models/genesis_config.py',
                    'Path to a model config file.')
# Trained model
flags.DEFINE_string('model_dir', 'checkpoints/test/1',
                    'Path to model directory.')
flags.DEFINE_string('model_file', 'model.ckpt-FINAL', 'Name of model file.')
# Other
flags.DEFINE_integer('num_images', 10, 'Number of images to visualize.')


def main():
    # Parse flags
    config = forge.config()
    fet.print_flags()
    # Restore flags of pretrained model
    flag_path = osp.join(config.model_dir, 'flags.json')
    fprint(f"Restoring flags from {flag_path}")
    pretrained_flags = AttrDict(fet.json_load(flag_path))
    pretrained_flags.debug = True

    # Fix seeds. Always first thing to be done after parsing the config!
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    config.batch_size = 1
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
    for count, batch in enumerate(test_loader):
        if count >= config.num_images:
            break

        # Forward pass
        output, _, stats, _, _ = model(batch['input'])
        # Set up figure
        fig, axes = plt.subplots(nrows=4, ncols=1+pretrained_flags.K_steps)     

        # Input and reconstruction
        plot(axes, 0, 0, batch['input'], title='Input image', fontsize=12)
        plot(axes, 1, 0, output, title='Reconstruction', fontsize=12)
        # Empty plots
        plot(axes, 2, 0, fontsize=12)
        plot(axes, 3, 0, fontsize=12)
        
        # Put K reconstruction steps into separate subfigures
        x_k = stats['x_r_k']
        log_m_k = stats['log_m_k']
        mx_k = [x*m.exp() for x, m in zip(x_k, log_m_k)]
        log_s_k = stats['log_s_k'] if 'log_s_k' in stats else None
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
        manager.resize(*manager.window.maxsize())
        plt.show()


if __name__ == "__main__":
    main()
