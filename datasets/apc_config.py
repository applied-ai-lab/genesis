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
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput


flags.DEFINE_string('data_folder', 'data/apc',
                    'Path to data folder.')
flags.DEFINE_integer('img_size', 128,
                     'Dimension of images. Images are square.')
flags.DEFINE_integer('num_workers', 4, 'Number of threads for loading data.')

flags.DEFINE_integer('K_steps', 10, 'Number of component steps.')


def load(cfg, **unused_kwargs):

    del unused_kwargs
    if not os.path.exists(cfg.data_folder):
        raise Exception("Data folder does not exist.")

    assert cfg.img_size == 128

    # Create splits if needed
    modes = ['train', 'val', 'test']
    create_splits = False
    for m in modes:
        if not os.path.exists(f'{cfg.data_folder}/{m}_images.txt'):
            create_splits = True
            break
    if create_splits:
        fprint("Creating new train/val/test splits...")
        # Randomly split into train/val/test with fixed seed
        all_scenes = sorted(glob(f'{cfg.data_folder}/processed/*/*/scene-*'))
        random.seed(0)
        random.shuffle(all_scenes)
        num_eval_scenes = len(all_scenes) // 10
        train_scenes = all_scenes[2*num_eval_scenes:]
        val_scenes = all_scenes[:num_eval_scenes]
        test_scenes = all_scenes[num_eval_scenes:2*num_eval_scenes]
        modes = ['train', 'val', 'test']
        mode_scenes = [train_scenes, val_scenes, test_scenes]
        for mode, mscs in zip(modes, mode_scenes):
            img_paths = []
            for sc in mscs:
                img_paths += glob(f'{sc}/frame-*.color.png')
            with open(f'{cfg.data_folder}/{mode}_images.txt', 'w') as f:
                for item in sorted(img_paths):
                    f.write("%s\n" % item)
        # Sanity checks
        assert len(train_scenes + val_scenes + test_scenes) == len(all_scenes)
        assert not list(set(train_scenes).intersection(val_scenes))
        assert not list(set(train_scenes).intersection(test_scenes))
        assert not list(set(val_scenes).intersection(test_scenes))
        fprint("Created new train/val/test splits!")

    # Read splits
    with open(f'{cfg.data_folder}/train_images.txt') as f:
        train_images = f.readlines()
        train_images = [x.strip() for x in train_images]
    with open(f'{cfg.data_folder}/val_images.txt') as f:
        val_images = f.readlines()
        val_images = [x.strip() for x in val_images]
    with open(f'{cfg.data_folder}/test_images.txt') as f:
        test_images = f.readlines()
        test_images = [x.strip() for x in test_images]
    fprint(f"{len(train_images)} train images")
    fprint(f"{len(val_images)} val images")
    fprint(f"{len(test_images)} test images")

    # Datasets
    trainset = APCDataset(train_images)
    valset = APCDataset(val_images)
    testset = APCDataset(test_images)
    # Loaders
    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers)
    val_loader = DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers)
    test_loader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers)

    # Throughput stats
    if not cfg.debug:
        loader_throughput(train_loader)

    return (train_loader, val_loader, test_loader)


class APCDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths
        # Transforms
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        fp = self.image_paths[idx]
        img = self.transform(Image.open(fp))
        mfp= fp.replace('frame', 'masks/frame').replace('color', 'mask')
        mask = self.transform(Image.open(mfp)).long()
        return {'images': img, 'instances': mask}


def preprocess(data_folder='data/apc', img_size=128):
    print("Getting image paths...")
    image_paths = glob(
        f'{data_folder}/training/*/*/scene-*/frame-*.color.png')
    print(f"Done. Found {len(image_paths)}.")
    img_T = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BILINEAR),
        transforms.CenterCrop(img_size)
    ])
    mask_T = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.NEAREST),
        transforms.CenterCrop(img_size)
    ])
    # Created folders
    print("Creating folders...")
    for path in tqdm(glob(f'{data_folder}/training/*/*/scene-*/')):
        os.makedirs(path.replace('training', 'processed'))
        os.makedirs(path.replace('training', 'processed')+'/masks')
    print("Done.")
    print("Preprocessing images...")
    # Preprocess images
    for path in tqdm(image_paths):
        # Image
        img = img_T(Image.open(path))
        img.save(path.replace('training', 'processed'))
        # Mask
        if 'scene-empty' not in path:
            m_path = path.replace('frame', 'masks/frame').replace('color', 'mask')
            mask = mask_T(Image.open(m_path))
            mask.save(m_path.replace('training', 'processed'))
    print("ALL DONE!")


if __name__ == "__main__":
    preprocess()
