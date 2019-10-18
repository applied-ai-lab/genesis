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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

import torch

import forge
from forge import flags
import forge.experiment_tools as fet

# Config
flags.DEFINE_string('data_config', 'datasets/multid_config.py',
                    'Path to a data config file.')
flags.DEFINE_integer('batch_size', 8, 'Mini-batch size.')
flags.DEFINE_integer('seed', 0, 'Seed for random number generators.')


def main():
    # Parse flags
    cfg = forge.config()
    cfg.num_workers = 0

    # Set manual seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get data loaders
    train_loader, _, _ = fet.load(cfg.data_config, cfg)

    # Visualise
    for x in train_loader:
        fig, axes = plt.subplots(1, cfg.batch_size, figsize=(20, 10))

        img = x['input']
        for b_idx in range(cfg.batch_size):
            np_img = np.moveaxis(img.data.numpy()[b_idx], 0, -1)
            if img.shape[1] == 1:
                axes[b_idx].imshow(
                    np_img[:, :, 0], norm=NoNorm(), cmap='gray')
            elif img.shape[1] == 3:
                axes[b_idx].imshow(np_img)
            axes[b_idx].axis('off')

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.show()

if __name__ == "__main__":
    main()
