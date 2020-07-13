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

import os
import time

import torch
import torch.nn.functional as F

import tensorflow as tf

import numpy as np

import third_party.tf_gqn.gqn_tfr_provider as gqn

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput


flags.DEFINE_string('data_folder', 'data/gqn_datasets',
                    'Path to data folder.')
flags.DEFINE_integer('img_size', 64,
                     'Dimension of images. Images are square.')
flags.DEFINE_integer('val_frac', 60,
                     'Fraction of training images to use for validation.')

flags.DEFINE_integer('num_workers', 4, 'TF records dataset.')
flags.DEFINE_integer('buffer_size', 128, 'TF records dataset.')

flags.DEFINE_integer('K_steps', 7, 'Number of recurrent steps.')


SEED = 0


def load(cfg, **unused_kwargs):
    # Fix TensorFlow seed
    global SEED
    SEED = cfg.seed
    tf.set_random_seed(SEED)

    if cfg.num_workers == 0:
        fprint("Need to use at least one worker for loading tfrecords.")
        cfg.num_workers = 1

    del unused_kwargs
    if not os.path.exists(cfg.data_folder):
        raise Exception("Data folder does not exist.")
    print(f"Using {cfg.num_workers} data workers.")
    # Create data iterators
    train_loader = GQNLoader(
        data_folder=cfg.data_folder, mode='devel_train', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, buffer_size=cfg.buffer_size)
    val_loader = GQNLoader(
        data_folder=cfg.data_folder, mode='devel_val', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, buffer_size=cfg.buffer_size)
    test_loader = GQNLoader(
        data_folder=cfg.data_folder, mode='test', img_size=cfg.img_size,
        val_frac=cfg.val_frac, batch_size=1,
        num_workers=1, buffer_size=cfg.buffer_size)
    # Create session to be used by loaders
    sess = tf.InteractiveSession()
    train_loader.sess = sess
    val_loader.sess = sess
    test_loader.sess = sess

    # Throughput stats
    loader_throughput(train_loader)

    return (train_loader, val_loader, test_loader)


class GQNLoader():
    """GQN dataset."""

    def __init__(self, data_folder, mode, img_size, val_frac, batch_size,
                 num_workers, buffer_size):
        self.img_size = img_size
        self.batch_size = batch_size
        self.sess = None
        # Create GQN reader
        reader = gqn.GQNTFRecordDataset(
            dataset='rooms_ring_camera',
            context_size=0,
            root=data_folder,
            mode=mode,
            val_frac=val_frac,
            custom_frame_size=None,
            num_threads=num_workers,
            buffer_size=buffer_size)
        # Operato on TFRecordsDataset
        dataset = reader._dataset
        # Set properties
        dataset = dataset.repeat(1)
        if 'train' in mode:
            dataset = dataset.shuffle(
                buffer_size=buffer_size * self.batch_size, seed=SEED)
        dataset = dataset.batch(self.batch_size)
        self.dataset = dataset.prefetch(buffer_size * self.batch_size)
        # Create iterator
        it = self.dataset.make_one_shot_iterator()
        self.frames, _ = it.get_next()
        # TODO(martin): avoid hard coding these
        train_sz = 10800000
        test_sz = 1200000
        if mode == 'train':
            num_frames = train_sz
        elif mode == 'test':
            num_frames = test_sz
        elif mode == 'devel_train':
            num_frames = (train_sz // val_frac) * (val_frac-1)
        elif mode == 'devel_val':
            num_frames = (train_sz // val_frac)
        else:
            raise ValueError("Mode not known.")
        self.length = num_frames // batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        try:
            img = self.sess.run(self.frames)
            img = img[:, 0, :, :, :]
            img = np.moveaxis(img, 3, 1)
            img = torch.FloatTensor(img)
            if self.img_size != 64:
                img = F.interpolate(img, size=self.img_size)
            return {'input': img}
        except tf.errors.OutOfRangeError:
            print("Reached end of epoch. Creating new iterator.")
            # Create new iterator for next epoch
            it = self.dataset.make_one_shot_iterator()
            self.frames, _ = it.get_next()
            raise StopIteration
