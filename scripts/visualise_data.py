# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import json
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

    # Optimally distinct RGB colour palette (15 colours)
    colours = json.load(open('utils/colour_palette15.json'))

    # Visualise
    for x in train_loader:
        fig, axes = plt.subplots(2, cfg.batch_size, figsize=(20,10))

        for f_idx, field in enumerate(['input', 'instances']):
            for b_idx in range(cfg.batch_size):
                axes[f_idx, b_idx].axis('off')

            if field not in x:
                continue
            img = x[field]

            # Colour instance masks
            if field == 'instances':
                img_list = []
                for b_idx in range(img.shape[0]):
                    instances = img[b_idx, :, :, :]
                    img_r = torch.zeros_like(instances)
                    img_g = torch.zeros_like(instances)
                    img_b = torch.zeros_like(instances)
                    ins_idx = 0
                    for ins in range(instances.max().numpy()):
                        ins_map = instances == ins + 1
                        if ins_map.any():
                            img_r[ins_map] = colours['palette'][ins_idx][0]
                            img_g[ins_map] = colours['palette'][ins_idx][1]
                            img_b[ins_map] = colours['palette'][ins_idx][2]
                            ins_idx += 1
                    img_list.append(torch.cat([img_r, img_g, img_b], dim=0))
                img = torch.stack(img_list, dim=0)

            for b_idx in range(cfg.batch_size):
                np_img = np.moveaxis(img.data.numpy()[b_idx], 0, -1)
                if img.shape[1] == 1:
                    axes[f_idx, b_idx].imshow(
                        np_img[:, :, 0], norm=NoNorm(), cmap='gray')
                elif img.shape[1] == 3:
                    axes[f_idx, b_idx].imshow(np_img)

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.show()

if __name__ == "__main__":
    main()
