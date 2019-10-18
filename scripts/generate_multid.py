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

import random
from random import randint, choice

import torch

import numpy as np
from PIL import Image


# Set manual seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def rand_rgb_tuble():
    val = [0, 63, 127, 191, 255]
    return choice(val), choice(val), choice(val)


def generate(sprites, dataset_size, num_objects=None):
    # Initialise
    all_images = np.zeros((dataset_size, 64, 64, 3))
    all_instance_masks = np.zeros((dataset_size, 64, 64, 1))

    # Create images
    for i in range(dataset_size):
        if (i+1)%10000 == 0:
            print(f"Processing [{i+1} | {dataset_size}]")

        # Create background
        image = np.array(Image.new('RGB', (64, 64), rand_rgb_tuble()))
        # Initialise instance masks
        instance_masks = np.zeros((64, 64, 1)).astype('int')

        # Add objects
        if num_objects is None:
            num_sprites = randint(1, 4)
        else:
            num_sprites = num_objects
        for obj_idx in range(num_sprites):
            object_index = randint(0, 737279)
            sprite_mask = np.array(sprites[object_index], dtype=bool)
            crop_index = np.where(sprite_mask == True)
            image[crop_index] = rand_rgb_tuble()
            instance_masks[crop_index] = obj_idx + 1
        # Collate
        all_images[i] = image
        all_instance_masks[i] = instance_masks

    all_images = all_images.astype('float32') / 255.0
    return all_images, all_instance_masks


def main():
    # Load dataset
    dataset_zip = np.load(
        'data/multi_dsprites/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        encoding="latin1")
    sprites = dataset_zip['imgs']

    # --- 1-4 objects ---

    # Generate training data
    print("Generate training images...")
    train_images, train_masks = generate(sprites, 50000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/training_images_rand4.npy", train_images)
    np.save("data/multi_dsprites/processed/training_masks_rand4.npy", train_masks)

    # Generate validation data
    print("Generate validation images...")
    val_images, val_masks = generate(sprites, 10000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/validation_images_rand4.npy", val_images)
    np.save("data/multi_dsprites/processed/validation_masks_rand4.npy", val_masks)

    # Generate test data
    print("Generate test images...")
    test_images, test_masks = generate(sprites, 10000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/test_images_rand4.npy", test_images)
    np.save("data/multi_dsprites/processed/test_masks_rand4.npy", test_masks)

    print("Done!")


if __name__ == "__main__":
    main()
