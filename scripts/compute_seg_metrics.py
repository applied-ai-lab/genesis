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
from attrdict import AttrDict
from tqdm import tqdm
import random

import torch

import numpy as np

import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

from utils.misc import average_ari, average_segcover


# Config
flags.DEFINE_string('data_config', 'datasets/shapestacks_config.py',
                    'Path to a data config file.')
flags.DEFINE_string('model_config', 'models/genesis_config.py',
                    'Path to a model config file.')
# Trained model
flags.DEFINE_string('model_dir', 'checkpoints/test/1',
                    'Path to model directory.')
flags.DEFINE_string('model_file', 'model.ckpt-FINAL', 'Name of model file.')
# Other
flags.DEFINE_integer('seed', 0, 'Seed for random number generators.')
flags.DEFINE_integer('num_images', 300, 'Number of images to run on.')
flags.DEFINE_string('split', 'test', '{train, val, test}')

# Set manual seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # Parse flags
    config = forge.config()
    config.batch_size = 1
    config.load_instances = True
    fet.print_flags()

    # Restore original model flags
    pretrained_flags = AttrDict(
        fet.json_load(os.path.join(config.model_dir, 'flags.json')))

    # Get validation loader
    train_loader, val_loader, test_loader = fet.load(config.data_config, config)
    fprint(f"Split: {config.split}")
    if config.split == 'train':
        batch_loader = train_loader
    elif config.split == 'val':
        batch_loader = val_loader
    elif config.split == 'test':
        batch_loader = test_loader
    # Shuffle and prefetch to get same data for different models
    if 'gqn' not in config.data_config:
        batch_loader = torch.utils.data.DataLoader(
            batch_loader.dataset, batch_size=1, num_workers=0, shuffle=True)
    # Prefetch batches
    prefetched_batches = []
    for i, x in enumerate(batch_loader):
        if i == config.num_images:
            break
        prefetched_batches.append(x)

    # Load model
    model = fet.load(config.model_config, pretrained_flags)
    fprint(model)
    model_path = os.path.join(config.model_dir, config.model_file)
    fprint(f"Restoring model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_1', None)
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_2', None)
    model.load_state_dict(model_state_dict)

    # Set experiment folder and fprint file for logging
    fet.EXPERIMENT_FOLDER = config.model_dir
    fet.FPRINT_FILE = 'segmentation_metrics.txt'

    # Compute metrics
    model.eval()
    ari_fg_list, sc_fg_list, msc_fg_list = [], [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(prefetched_batches)):
            _, _, stats, _, _ = model(x['input'])
            # ARI
            ari_fg, _ = average_ari(stats.log_m_k, x['instances'],
                                    foreground_only=True)
            # Segmentation covering - foreground only
            gt_instances = x['instances'].clone()
            gt_instances[gt_instances == 0] = -100
            ins_preds = torch.argmax(torch.stack(stats.log_m_k, dim=1), dim=1)
            sc_fg = average_segcover(gt_instances, ins_preds)
            msc_fg = average_segcover(gt_instances, ins_preds, False)
            # Recording
            ari_fg_list.append(ari_fg)
            sc_fg_list.append(sc_fg)
            msc_fg_list.append(msc_fg)

    # Print average metrics
    fprint(f"Average FG ARI: {sum(ari_fg_list)/len(ari_fg_list)}")
    fprint(f"Average FG SegCover: {sum(sc_fg_list)/len(sc_fg_list)}")
    fprint(f"Average FG MeanSegCover: {sum(msc_fg_list)/len(msc_fg_list)}")


if __name__ == "__main__":
    main()
