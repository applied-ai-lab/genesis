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

import sys
import os
from os import path as osp
import random
import time
import datetime

from attrdict import AttrDict

import torch
from torch.utils.data.dataset import TensorDataset

import numpy as np
from PIL import Image

from tqdm import tqdm

import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

from third_party.pytorch_fid import fid_score as FID


def main_flags():
    # Data & model config
    flags.DEFINE_string('data_config', 'datasets/gqn_config.py',
                        'Path to a data config file.')
    flags.DEFINE_string('model_config', 'models/genesis_config.py',
                        'Path to a model config file.')
    # Pre-trained model
    flags.DEFINE_string('model_dir', 'checkpoints/test/1',
                        'Path to model directory.')
    flags.DEFINE_string('model_file', 'model.ckpt-FINAL', 'Name of model file.')
    # FID
    flags.DEFINE_integer('feat_dim', 2048, 'Number of Incpetion features.')
    flags.DEFINE_integer('num_fid_images', 10000,
                         'Number of images to compute the FID on.')
    # Other
    flags.DEFINE_string('img_dir', '/tmp', 'Directory for saving pngs.')
    flags.DEFINE_integer('batch_size', 10, 'Mini-batch size.')
    flags.DEFINE_boolean('gpu', True, 'Use GPU if available.')
    flags.DEFINE_integer('seed', 0, 'Seed for random number generators.')


def main():
    # Parse flags
    config = forge.config()
    fet.EXPERIMENT_FOLDER = config.model_dir
    fet.FPRINT_FILE = 'fid_evaluation.txt'
    config.shuffle_test = True

    # Fix seeds. Always first thing to be done after parsing the config!
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Using GPU?
    if torch.cuda.is_available() and config.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        config.gpu = False
        torch.set_default_tensor_type('torch.FloatTensor')
    fet.print_flags()

    # Load data
    if config.compute_fid:
        _, _, test_loader = fet.load(config.data_config, config)

    #  Load model
    flag_path = osp.join(config.model_dir, 'flags.json')
    fprint(f"Restoring flags from {flag_path}")
    pretrained_flags = AttrDict(fet.json_load(flag_path))
    model = fet.load(config.model_config, pretrained_flags)
    model_path = osp.join(config.model_dir, config.model_file)
    fprint(f"Restoring model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_1', None)
    model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_2', None)
    model.load_state_dict(model_state_dict)
    fprint(model)
    # Put model on GPU
    if config.gpu:
        model = model.cuda()

    # Compute FID
    fid_from_model(model, test_loader, config.batch_size,
                   config.num_fid_images, config.feat_dim, config.img_dir)


def fid_from_model(model, test_loader, batch_size=10, num_images=10000,
                   feat_dim=2048, img_dir='/tmp'):

    model.eval()

    # Save images from test set as pngs
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fprint(t + " | Saving images from test set as pngs.")
    test_dir = osp.join(img_dir, 'test_images')
    os.makedirs(test_dir)
    count = 0
    for bidx, batch in enumerate(test_loader):
        count = tensor_to_png(batch['input'], test_dir, count, num_images)
        if count >= num_images:
            break

    # Generate images and save as pngs
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fprint(t + " | Generate images and save as pngs.")
    gen_dir = osp.join(img_dir, 'generated_images')
    os.makedirs(gen_dir)
    count = 0
    for _ in tqdm(range(num_images // batch_size + 1)):
        if count >= num_images:
            break
        with torch.no_grad():
            gen_img, _ = model.sample(batch_size)
        count = tensor_to_png(gen_img, gen_dir, count, num_images)

    # Compute FID
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fprint(t + " | Computing FID.")
    gpu = next(model.parameters()).is_cuda
    fid_value = FID.calculate_fid_given_paths(
        [test_dir, gen_dir], batch_size, gpu, feat_dim)
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fprint(t + f" | FID: {fid_value}")

    model.train()

    return fid_value


def tensor_to_png(tensor, save_dir, count, stop):
    np_images = tensor.cpu().numpy()
    np_images = np.moveaxis(np_images, 1, 3)
    for i in range(len(np_images)):
        im = Image.fromarray(np.uint8(255*np_images[i]))
        fn = osp.join(save_dir, str(count).zfill(6) + '.png')
        im.save(fn)
        count += 1
        if count >= stop:
            return count
    return count


if __name__ == "__main__":
    main_flags()
    main()
