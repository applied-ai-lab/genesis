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
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput


flags.DEFINE_string('data_folder', 'data/sketchy', 'Path to data folder.')
flags.DEFINE_integer('num_workers', 4, 'Number of threads for loading data.')
flags.DEFINE_integer('img_size', 128, 'Dimension of images. Images are square.')
# Object slots: 3 objects, robot base, gripper, wrist, arm, ground, cables, wall
flags.DEFINE_integer('K_steps', 10, 'Number of object slots.')


def load(cfg, **unused_kwargs):
    del unused_kwargs
    if not os.path.exists(cfg.data_folder):
        raise Exception("Data folder does not exist.")
    fprint(f"Using {cfg.num_workers} data workers.")

    assert cfg.img_size == 128

    tng_set = SketchyDataset(cfg.data_folder, 'train')
    val_set = SketchyDataset(cfg.data_folder, 'valid')
    tst_set = SketchyDataset(cfg.data_folder, 'test')

    tng_loader = DataLoader(
        tng_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)
    tst_loader = DataLoader(
        tst_set,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    if not cfg.debug:
        loader_throughput(tng_loader)

    return tng_loader, val_loader, tst_loader


class SketchyDataset(Dataset):

    def __init__(self, data_dir, mode):
        split_file = f'{data_dir}/processed/{mode}_images.txt'
        if os.path.exists(split_file):
            fprint(f"Reading paths for {mode} files...")
            with open(split_file, "r") as f:
                self.filenames = f.readlines()
            self.filenames = [item.strip() for item in self.filenames]
        else:
            fprint(f"Searching for {mode} files...")
            self.filenames = glob(f'{data_dir}/processed/{mode}/ep*/ep*.png')
            with open(split_file, "w") as f:
                for item in self.filenames:
                    f.write(f'{item}\n')
        fprint(f"Found {len(self.filenames)}.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        img = Image.open(file)
        return {'images': transforms.functional.to_tensor(img)}
