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
from tqdm import tqdm
from PIL import Image

import torch
from sketchy import sketchy

data_folder = 'data/sketchy'
filenames = sorted(glob(f'{data_folder}/records/*'))

# Split into train/valid/test files
num_files = len(filenames)
num_eval = num_files//10
valid_files = filenames[:num_eval]
test_files = filenames[num_eval:2*num_eval]
train_files = filenames[2*num_eval:]

thumbnail_size = (128, 128)

# Check for dublicates
all_files = train_files + valid_files + test_files
assert not len(all_files) != len(set(all_files))

episode_idx = 0
for mode, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
    save_folder = f'{data_folder}/processed/{mode}'
    print(f'Processing {mode} data. Destination: {save_folder}')
    os.makedirs(save_folder)
    for episode_file in tqdm(files):
        episode = sketchy.load_frames(episode_file, 4)
        episode_folder = f'{save_folder}/ep{str(episode_idx).zfill(6)}'
        os.makedirs(episode_folder)
        prefix = f'{episode_folder}/ep{str(episode_idx).zfill(6)}'
        for ex_idx, frame in enumerate(episode):
            im_fl = frame['pixels/basket_front_left'].numpy()
            im_fr = frame['pixels/basket_front_right'].numpy()
            # Crop to 448x672            
            im_fl = im_fl[71:-81, 144:-144]
            im_fr = im_fr[91:-61, 144:-144]
            assert im_fl.shape == im_fr.shape
            ss = im_fl.shape[0]  # short side
            ls = im_fl.shape[1]  # long side
            cs = ss-64-32        # crop size
            mc = int(ls//2 - cs//2)  # middle crop location
            for im, view in zip([im_fl, im_fr], ['fl', 'fr']):
                # Save full image (448x448 crop)
                full = Image.fromarray(im[:, int(ls//2-ss//2):int(ls//2-ss//2)+ss])
                full = full.resize(thumbnail_size, resample=Image.BILINEAR)
                full.save(f'{prefix}_t{str(ex_idx).zfill(3)}_{view}_full.png')
                # Save crops
                c = 0
                for x1, x2 in zip([0, -cs], [cs, ss+1]):
                    for y1, y2 in zip([0, mc, -cs], [cs, mc+cs, ls+1]):
                        crop = im[x1:x2, y1:y2, :]
                        crop = Image.fromarray(crop)
                        crop = crop.resize(thumbnail_size, resample=Image.BILINEAR)
                        crop.save(f'{prefix}_t{str(ex_idx).zfill(3)}_{view}_c{c}.png')
                        c += 1
            state = {}
            for key, val in frame.items():
                if 'pixels' in key:
                    continue
                state[key] = torch.tensor(val.numpy())
            torch.save(state, f'{prefix}_t{str(ex_idx).zfill(3)}_state.pt')
        episode_idx += 1
