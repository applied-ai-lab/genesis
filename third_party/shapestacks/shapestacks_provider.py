"""
Provides an input_fn for tf.estimator.Estimator to load the images of the real
synthetic simulation recordings of a ShapeStacks dataset.

Adapted from https://github.com/ogroth/shapestacks

Modified by Martin Engelcke
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import tensorflow as tf
# import numpy as np


# dataset constants
_CHANNELS = 3 # RGB images
_HEIGHT = 224
_WIDTH = 224
_NUM_CLASSES = 2 # stable | unstable
# label semantics: 0 = stable | 1 = unstable

# data augmentation constants
_CROP_HEIGHT = 196
_CROP_WIDTH = 196


# internal dataset creation, file parsing and pre-processing

def _get_filenames_with_labels(mode, data_dir, split_dir):
  """
  Returns all training or test files in the data directory with their
  respective labels.
  """
  if mode == 'train':
    scenario_list_file = os.path.join(split_dir, 'train.txt')
  elif mode == 'eval':
    scenario_list_file = os.path.join(split_dir, 'eval.txt')
  elif mode == 'test':
    scenario_list_file = os.path.join(split_dir, 'test.txt')
  else:
    raise ValueError("Mode %s is not supported!" % mode)
  with open(scenario_list_file) as f:
    scenario_list = f.read().split('\n')
    scenario_list.pop()

  filenames = []
  labels = []
  for i, scenario in enumerate(scenario_list):
    if (i+1) % 100 == 0:
      print("%s / %s : %s" % (i+1, len(scenario_list), scenario))
    scenario_dir = os.path.join(data_dir, 'recordings', scenario)
    if "vcom=0" in scenario and "vpsf=0" in scenario: # stable scenario
      label = 0.0
    else: # unstable scenario
      label = 1.0
    for img_file in filter(
        lambda f: f.startswith('rgb-') and f.endswith('-mono-0.png'),
        os.listdir(scenario_dir)):
      filenames.append(os.path.join(scenario_dir, img_file))
      labels.append(label)

  return filenames, labels

# def _create_dataset(filenames, labels):
#   """
#   Creates a dataset from the given filename and label tensors.
#   """
#   tf_filenames = tf.constant(filenames)
#   tf_labels = tf.constant(labels)
#   dataset = tf.data.Dataset.from_tensor_slices((tf_filenames, tf_labels))
#   return dataset

# def _parse_record(filename, label):
#   """
#   Reads the file and returns a (feature, label) pair.
#   Image feature values are returned to scale in [0.0, 1.0].
#   """
#   image_string = tf.read_file(filename)
#   image_decoded = tf.image.decode_image(image_string, channels=_CHANNELS)
#   image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, _HEIGHT, _WIDTH)
#   image_float = tf.cast(image_resized, tf.float32)
#   image_float = tf.reshape(image_float, [_HEIGHT, _WIDTH, _CHANNELS])
#   return image_float, label

# def _augment(feature, label, augment):
#   """
#   Applies data augmentation to the features.
#   Augmentaion contains:
#   - random cropping and resizing back to _HEIGHT & _WIDTH
#   - random LR flip
#   - random recoloring
#   - clip within [-1, 1]
#   """

#   feature = tf.image.convert_image_dtype(feature, tf.float32, saturate=True)
#   convert_factor = 1.0

#   if 'rotate' in augment:
#     random_rotation = tf.reshape(
#         tf.random_uniform([1], minval=-0.01, maxval=0.01, dtype=tf.float32),
#         [])
#     feature = tf.contrib.image.rotate(
#         feature, random_rotation * 3.1415, interpolation='BILINEAR')

#   if 'convert' in augment:
#     feature = tf.multiply(feature, 1.0 / 255.0)
#     convert_factor = 255.0

#   if 'crop' in augment:

#     if 'stretch' in augment:
#       rand_crop_height = tf.reshape(
#           tf.random_uniform(
#               [1], minval=_CROP_HEIGHT, maxval=_HEIGHT, dtype=tf.int32),
#           [])
#       rand_crop_width = tf.reshape(
#           tf.random_uniform(
#               [1], minval=_CROP_WIDTH, maxval=_WIDTH, dtype=tf.int32),
#           [])
#     else:
#       rand_crop_height = _CROP_HEIGHT
#       rand_crop_width = _CROP_WIDTH

#     feature = tf.random_crop(
#         value=feature, size=[rand_crop_height, rand_crop_width, _CHANNELS])
#     feature = tf.image.resize_bilinear(
#         images=tf.reshape(
#             feature, [1, rand_crop_height, rand_crop_width, _CHANNELS]),
#         size=[_HEIGHT, _WIDTH])

#   if 'flip' in augment:
#     feature = tf.image.random_flip_left_right(
#         tf.reshape(feature, [_HEIGHT, _WIDTH, _CHANNELS]))

#   if 'recolour' in augment:
#     feature = tf.image.random_brightness(feature, max_delta=32. / convert_factor)
#     feature = tf.image.random_saturation(feature, lower=0.5, upper=1.5)
#     feature = tf.image.random_hue(feature, max_delta=0.2)
#     feature = tf.image.random_contrast(feature, lower=0.5, upper=1.5)

#   if 'noise' in augment:
#     # add gaussian noise
#     gaussian_noise = tf.random_normal(
#         [_HEIGHT, _WIDTH, _CHANNELS], stddev=4. / convert_factor)
#     feature = tf.add(feature, gaussian_noise)

#   if 'clip' in augment:
#     if 'convert' in augment:
#       # clip to [0,1]
#       feature = tf.clip_by_value(feature, 0.0, 1.0)
#     else:
#       feature = tf.clip_by_value(feature, 0.0, 255.0)

#   if 'center' in augment:
#     # center around 0
#     feature = tf.subtract(feature, 0.5)
#     feature = tf.multiply(feature, 2.0)

#   feature = tf.reshape(feature, [_HEIGHT, _WIDTH, _CHANNELS])
#   return feature, label

# def _center_data(feature, label, rgb_mean):
#   """
#   Subtracts the mean of the respective data split part to center the data.
#   rgb_mean is expected to scale in [0.0, 1.0].
#   """
#   feature_centered = feature - tf.reshape(tf.constant(rgb_mean), [1, 1, 3])
#   return feature_centered, label


# # public input_fn for dataset iteration

# def shapestacks_input_fn(
#     mode, data_dir, split_name,
#     batch_size, num_epochs=1,
#     n_prefetch=2, augment=[]):
#   """
#   Input_fn to feed a tf.estimator.Estimator with ShapeStacks images.

#   Args:
#     mode: train | eval | test
#     data_dir:
#     split_name: directory name under data_dir/splits containing train.txt, eval.txt and test.txt
#     batch_size:
#     num_epochs:
#     n_prefetch: number of images to prefetch into RAM
#     augment: data augmentations to apply
#       'rotate': randomly rotates the image in plane by +/- 2 degrees
#       'convert': converts input values into [0.0, 1.0]
#       'crop': performs a random quadratic center crop
#       'stretch': performs a random center crop not preserving aspect ratio
#       'flip': applies a random left-right flip
#       'recolour': recolours the image by randomly tuning brightness, saturation,
#         hue and contrast
#       'noise': adds Gaussian noise to the image
#       'clip': clips input values to [0.0, 1.0]
#       'center':
#       'subtract_mean': subtracts the RGB mean of the data chunk loaded
#   """
#   split_dir = os.path.join(data_dir, 'splits', split_name)
#   filenames, labels = _get_filenames_with_labels(mode, data_dir, split_dir)
#   rgb_mean_npy = np.load(os.path.join(split_dir, mode + '_bgr_mean.npy'))[[2, 1, 0]]
#   dataset = _create_dataset(filenames, labels)

#   # shuffle before providing data
#   if mode == 'train':
#     dataset = dataset.shuffle(buffer_size=len(filenames))

#   # parse data from files and apply pre-processing
#   dataset = dataset.map(_parse_record)
#   if augment != [] and mode == 'train':
#     dataset = dataset.map(lambda feature, label: _augment(feature, label, augment))
#   if 'subtract_mean' in augment:
#     dataset = dataset.map(lambda feature, label: _center_data(feature, label, rgb_mean_npy))

#   # prepare batch and epoch cycle
#   dataset = dataset.prefetch(n_prefetch * batch_size)
#   dataset = dataset.repeat(num_epochs)
#   dataset = dataset.batch(batch_size)

#   # set up iterator
#   iterator = dataset.make_one_shot_iterator()
#   images, labels = iterator.get_next()
#   return images, labels