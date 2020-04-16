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
from shutil import copytree

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput, np_img_centre_crop

from third_party.shapestacks.shapestacks_provider import _get_filenames_with_labels
from third_party.shapestacks.segmentation_utils import load_segmap_as_matrix


flags.DEFINE_string('data_folder', 'data/shapestacks', 'Path to data folder.')
flags.DEFINE_string('split_name', 'default', '{default, blocks_all, css_all}')
flags.DEFINE_integer('img_size', 64, 'Dimension of images. Images are square.')
flags.DEFINE_boolean('shuffle_test', False, 'Shuffle test set.')

flags.DEFINE_integer('num_workers', 4, 'Number of threads for loading data.')
flags.DEFINE_boolean('load_instances', False, 'Load instances.')
flags.DEFINE_boolean('copy_to_tmp', False, 'Copy files to /tmp.')

flags.DEFINE_integer('K_steps', 9, 'Number of recurrent steps.')


MAX_SHAPES = 6
CENTRE_CROP = 196


def load(cfg, **unused_kwargs):
    del unused_kwargs
    if not os.path.exists(cfg.data_folder):
        raise Exception("Data folder does not exist.")
    print(f"Using {cfg.num_workers} data workers.")

    # Copy all images and splits to /tmp
    if cfg.copy_to_tmp:
        for directory in ['/recordings', '/splits', '/iseg']:
            src = cfg.data_folder + directory
            dst = '/tmp' + directory
            fprint(f"Copying dataset from {src} to {dst}.")
            copytree(src, dst)
        cfg.data_folder = '/tmp'

    # Training
    tng_set = ShapeStacksDataset(cfg.data_folder,
                                 cfg.split_name,
                                 'train',
                                 cfg.img_size,
                                 cfg.load_instances)
    tng_loader = DataLoader(tng_set,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers)
    # Validation
    val_set = ShapeStacksDataset(cfg.data_folder,
                                 cfg.split_name,
                                 'eval',
                                 cfg.img_size,
                                 cfg.load_instances)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers)
    # Test
    tst_set = ShapeStacksDataset(cfg.data_folder,
                                 cfg.split_name,
                                 'test',
                                 cfg.img_size,
                                 cfg.load_instances,
                                 shuffle_files=cfg.shuffle_test)
    tst_loader = DataLoader(tst_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)

    # Throughput stats
    loader_throughput(tng_loader)

    return (tng_loader, val_loader, tst_loader)


class ShapeStacksDataset(Dataset):

    def __init__(self, data_dir, split_name, mode, img_size=224,
                 load_instances=True, shuffle_files=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.load_instances = load_instances

        # Files
        split_dir = os.path.join(data_dir, 'splits', split_name)
        self.filenames, self.stability_labels = _get_filenames_with_labels(
            mode, data_dir, split_dir)

        # Shuffle files?
        if shuffle_files:
            print(f"Shuffling {len(self.filenames)} files")
            idx = np.arange(len(self.filenames), dtype='int32')
            np.random.shuffle(idx)
            self.filenames = [self.filenames[i] for i in list(idx)]
            self.stability_labels = [self.stability_labels[i] for i in list(idx)]

        # Transforms
        T = [transforms.CenterCrop(CENTRE_CROP)]
        if img_size != CENTRE_CROP:
            T.append(transforms.Resize(img_size))
        T.append(transforms.ToTensor())
        self.transform = transforms.Compose(T)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # --- Load image ---
        # File name example:
        # data_dir + /recordings/env_ccs-hard-h=2-vcom=0-vpsf=0-v=60/
        # rgb-w=5-f=2-l=1-c=unique-cam_7-mono-0.png
        file = self.filenames[idx]
        img = Image.open(file)
        output = {'input': self.transform(img)}

        # --- Load instances ---
        if self.load_instances:
            file_split = file.split('/')
            cam = file_split[4].split('-')[5][4:]
            map_path = os.path.join(
                self.data_dir, 'iseg', file_split[3],
                'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
            masks = load_segmap_as_matrix(map_path)
            masks = np.expand_dims(masks, 0)
            masks = np_img_centre_crop(masks, CENTRE_CROP)
            masks = torch.FloatTensor(masks)
            if self.img_size != masks.shape[2]:
                masks = masks.unsqueeze(0)
                masks = F.interpolate(masks, size=self.img_size)
                masks = masks.squeeze(0)
            output['instances'] = masks.type(torch.LongTensor)

        return output
