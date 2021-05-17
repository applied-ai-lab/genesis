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
flags.DEFINE_integer('num_images', 320, 'Number of images to run on.')
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
    config.debug = False
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
    ari_fg_list, msc_fg_list, ari_fg_r_list, msc_fg_r_list = [], [], [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(prefetched_batches)):
            _, _, stats, _, _ = model(x['input'])
            for mode in ['log_m_k', 'log_m_r_k']:
                if mode in stats:
                    log_masks = stats[mode]
                else:
                    continue
                # ARI
                ari_fg, _ = average_ari(log_masks, x['instances'],
                                        foreground_only=True)
                # Segmentation covering - foreground only
                ins_seg = torch.argmax(torch.cat(log_masks, 1), 1, True)
                msc_fg, _ = average_segcover(x['instances'], ins_seg, True)
                # Recording
                if mode == 'log_m_k':
                    ari_fg_list.append(ari_fg)
                    msc_fg_list.append(msc_fg)
                elif mode == 'log_m_r_k':
                    ari_fg_r_list.append(ari_fg)
                    msc_fg_r_list.append(msc_fg)

    # Print average metrics
    fprint(f"Average FG ARI: {sum(ari_fg_list)/len(ari_fg_list)}")
    fprint(f"Average FG MSC: {sum(msc_fg_list)/len(msc_fg_list)}")
    if len(ari_fg_r_list) > 0 and len(msc_fg_r_list) > 0:
        fprint(f"Average FG-R ARI: {sum(ari_fg_r_list)/len(ari_fg_r_list)}")
        fprint(f"Average FG-R MSC: {sum(msc_fg_r_list)/len(msc_fg_r_list)}")


if __name__ == "__main__":
    main()
