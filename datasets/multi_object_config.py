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

import torch
import torch.nn.functional as F

import tensorflow as tf

import numpy as np

from forge import flags
from forge.experiment_tools import fprint

from utils.misc import loader_throughput, len_tfrecords, np_img_centre_crop

import third_party.multi_object_datasets.multi_dsprites as multi_dsprites
import third_party.multi_object_datasets.objects_room as objects_room
import third_party.multi_object_datasets.clevr_with_masks as clevr_with_masks
import third_party.multi_object_datasets.tetrominoes as tetrominoes


flags.DEFINE_string('data_folder', 'data/multi-object-datasets',
                    'Path to data folder.')
flags.DEFINE_string('dataset', 'objects_room',
                    '{multi_dsprites, objects_room, clevr, tetrominoes}')
flags.DEFINE_integer('img_size', -1,
                     'Dimension of images. Images are square.')
flags.DEFINE_integer('dataset_size', -1, 'Number of images to use.')

flags.DEFINE_integer('num_workers', 4,
                     'Number of threads for loading data.')
flags.DEFINE_integer('buffer_size', 128, 'TF records dataset.')

flags.DEFINE_integer('K_steps', -1, 'Number of recurrent steps.')


MULTI_DSPRITES = '/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords'
OBJECTS_ROOM = '/objects_room/objects_room_train.tfrecords'
CLEVR = '/clevr_with_masks/clevr_with_masks_train.tfrecords'
TETROMINOS = '/tetrominoes/tetrominoes_train.tfrecords'
CLEVR_CROP = 192  # Following pre-processing in the IODINE paper

SEED = 0


def load(cfg, **unused_kwargs):
    # Fix TensorFlow seed
    global SEED
    SEED = cfg.seed
    tf.set_random_seed(SEED)

    del unused_kwargs
    fprint(f"Using {cfg.num_workers} data workers.")

    sess = tf.InteractiveSession()

    if cfg.dataset == 'multi_dsprites':
        cfg.img_size = 64 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 5 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 1
        max_frames = 60000
        raw_dataset = multi_dsprites.dataset(
            cfg.data_folder + MULTI_DSPRITES,
            'colored_on_colored',
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    elif cfg.dataset == 'objects_room':
        cfg.img_size = 64 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 7 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 4
        max_frames = 1000000
        raw_dataset = objects_room.dataset(
            cfg.data_folder + OBJECTS_ROOM,
            'train',
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    elif cfg.dataset == 'clevr':
        cfg.img_size = 128 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 11 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 1
        max_frames = 70000
        raw_dataset = clevr_with_masks.dataset(
            cfg.data_folder + CLEVR,
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    elif cfg.dataset == 'tetrominoes':
        cfg.img_size = 32 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 4 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 1
        max_frames = 60000
        raw_dataset = tetrominoes.dataset(
            cfg.data_folder + TETROMINOS,
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    else:
        raise NotImplementedError(f"{cfg.dataset} not a valid dataset.")

    # Split into train / val / test
    if cfg.dataset_size > max_frames:
        fprint(f"WARNING: {cfg.dataset_size} frames requested, "\
                "but only {max_frames} available.")
        cfg.dataset_size = max_frames
    if cfg.dataset_size > 0:
        total_sz = cfg.dataset_size
        raw_dataset = raw_dataset.take(total_sz)
    else:
        total_sz = max_frames
    if total_sz < 0:
        fprint("Determining size of dataset...")
        total_sz = len_tfrecords(raw_dataset, sess)
    fprint(f"Dataset has {total_sz} frames")
    
    val_sz = 10000
    tst_sz = 10000
    tng_sz = total_sz - val_sz - tst_sz
    assert tng_sz > 0
    fprint(f"Splitting into {tng_sz}/{val_sz}/{tst_sz} for tng/val/tst")
    tst_dataset = raw_dataset.take(tst_sz)
    val_dataset = raw_dataset.skip(tst_sz).take(val_sz)
    tng_dataset = raw_dataset.skip(tst_sz + val_sz)

    tng_loader = MultiOjectLoader(sess, tng_dataset, background_entities,
                                  tng_sz, cfg.batch_size,
                                  cfg.img_size, cfg.buffer_size)
    val_loader = MultiOjectLoader(sess, val_dataset, background_entities,
                                  val_sz, cfg.batch_size,
                                  cfg.img_size, cfg.buffer_size)
    tst_loader = MultiOjectLoader(sess, tst_dataset, background_entities,
                                  tst_sz, cfg.batch_size,
                                  cfg.img_size, cfg.buffer_size)

    # Throughput stats
    if not cfg.debug:
        loader_throughput(tng_loader)

    return (tng_loader, val_loader, tst_loader)


class MultiOjectLoader():

    def __init__(self, sess, dataset, background_entities,
                 num_frames, batch_size, img_size=64, buffer_size=128):
        # Batch and shuffle
        dataset = dataset.shuffle(buffer_size*batch_size, seed=SEED)
        dataset = dataset.batch(batch_size)
        self.dataset = dataset.prefetch(buffer_size)
        # State
        self.sess = sess
        self.background_entities = background_entities
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.length = self.num_frames // batch_size
        self.img_size = img_size
        self.count = 0
        self.frames = None

    def __len__(self):
        return self.length

    def __iter__(self):
        fprint("Creating new one_shot_iterator.")
        it = self.dataset.make_one_shot_iterator()
        self.frames = it.get_next()
        return self

    def __next__(self):
        try:
            frame = self.sess.run(self.frames)
            self.count += 1

            # Parse image
            img = frame['image']
            img = np.moveaxis(img, 3, 1)
            shape = img.shape
            # TODO(martin): use more explicit CLEVR flag?
            if shape[2] != shape[3]:
                img = np_img_centre_crop(img, CLEVR_CROP, batch=True)
            img = torch.FloatTensor(img) / 255.
            if self.img_size != shape[2]:
                img = F.interpolate(img, size=self.img_size)

            # Parse masks
            raw_masks = frame['mask']
            masks = np.zeros((shape[0], 1, shape[2], shape[3]), dtype='int')
            # Convert to boolean masks
            cond = np.where(raw_masks[:, :, :, :, 0] == 255, True, False)
            # Ignore background entities
            num_entities = cond.shape[1]
            for o_idx in range(self.background_entities, num_entities):
                masks[cond[:, o_idx:o_idx+1, :, :]] = o_idx + 1
            masks = torch.FloatTensor(masks)
            if shape[2] != shape[3]:
                masks = np_img_centre_crop(masks, CLEVR_CROP, batch=True)
            masks = torch.FloatTensor(masks)
            if self.img_size != shape[2]:
                masks = F.interpolate(masks, size=self.img_size)
            masks = masks.type(torch.LongTensor)

            return {'input': img, 'instances': masks}

        except tf.errors.OutOfRangeError:
            fprint("Reached end of epoch. Creating new iterator.")
            fprint(f"Counted {self.count} batches, expected {self.length}.")
            fprint("Creating new iterator.")
            self.count = 0
            raise StopIteration
