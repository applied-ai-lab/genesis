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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import numpy as np

from forge import flags

from utils.misc import loader_throughput


flags.DEFINE_string('data_folder', 'data/multi_dsprites/processed',
                    'Path to data folder.')
flags.DEFINE_boolean('unique_colours', False, 'Dataset with unique colours.')
flags.DEFINE_boolean('load_instances', True, 'Load instances.')
flags.DEFINE_integer('img_size', 64,
                     'Dimension of images. Images are square.')

flags.DEFINE_integer('num_workers', 4,
                     'Number of threads for loading data.')
flags.DEFINE_boolean('mem_map', False, 'Use memory mapping.')

flags.DEFINE_integer('K_steps', 5, 'Number of recurrent steps.')


def load(cfg, **unused_kwargs):
    """
    Args:
        cfg (obj): Forge config
    Returns:
        (DataLoader, DataLoader, DataLoader):
            Tuple of data loaders for train, val, test
    """
    del unused_kwargs
    if not os.path.exists(cfg.data_folder):
        raise Exception("Data folder does not exist.")
    print(f"Using {cfg.num_workers} data workers.")

    if not hasattr(cfg, 'unique_colours'):
        cfg.unique_colours = False

    # Paths
    if cfg.unique_colours:    
        train_path = 'training_images_rand4_unique.npy'
        val_path = 'validation_images_rand4_unique.npy'
        test_path = 'test_images_rand4_unique.npy'
    else:
        train_path = 'training_images_rand4.npy'
        val_path = 'validation_images_rand4.npy'
        test_path = 'test_images_rand4.npy'

    # Training
    train_dataset = dSpritesDataset(os.path.join(cfg.data_folder, train_path),
                                    cfg.load_instances,
                                    cfg.img_size,
                                    cfg.mem_map)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    # Validation
    val_dataset = dSpritesDataset(os.path.join(cfg.data_folder, val_path),
                                  cfg.load_instances,
                                  cfg.img_size,
                                  cfg.mem_map)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers)
    # Test
    test_dataset = dSpritesDataset(os.path.join(cfg.data_folder, test_path),
                                   cfg.load_instances,
                                   cfg.img_size,
                                   cfg.mem_map)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=1)

    # Throughput stats
    if not cfg.debug:
        loader_throughput(train_loader)

    return (train_loader, val_loader, test_loader)


class dSpritesDataset(Dataset):
    """dSprites dataset."""

    def __init__(self, file_path, load_instances=True,
                 img_size=64, mem_map=False):
        """
        Args:
            file_path (string): Path to the npy file of dSprites dataset.
            transform (callable, optional): Optional transform to be applied
        """
        if mem_map:
            self.all_images = np.load(file_path, mmap_mode='r')
        else:
            self.all_images = np.load(file_path)
        self.to_tensor = transforms.ToTensor()
        if load_instances and mem_map:
            self.all_instance_masks = np.load(
                file_path.replace('images', 'masks'), mmap_mode='r')
        elif load_instances:
            self.all_instance_masks = np.load(
                file_path.replace('images', 'masks'))
        else:
            self.all_instance_masks = None
        self.img_size = img_size

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img = self.all_images[idx]
        img = self.to_tensor(img)
        if self.img_size != 64:
            img = F.interpolate(img.unsqueeze(0), size=self.img_size).squeeze(0)
        output = {'input': img}
        if self.all_instance_masks is not None:
            ins = self.all_instance_masks[idx]
            ins = self.to_tensor(ins)
            if self.img_size != 64:
                ins = F.interpolate(
                    ins.unsqueeze(0), size=self.img_size).squeeze(0)
            output['instances'] = ins.type(torch.LongTensor)
        return output
