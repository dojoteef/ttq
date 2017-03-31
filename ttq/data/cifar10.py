# Copyright 2017 The Nader Akoury. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" The script to convert CIFAR-10 data to tf.train.Examples """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tarfile

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn
import tensorflow.contrib.slim as slim

import ttq.data.encode as encode_utils


NUM_CLASSES = 10
IMAGE_SHAPE = [32, 32, 3]
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32, 32, 3] image representing one of 10 classifications of object.',
    'label': 'An integer between [0 - 9] denoting image classification',
}
_IMAGES_PER_FILE = 10000
_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/'
_DATAFILE = 'cifar-10-binary.tar.gz'
_DATA_DIRECTORY = 'cifar-10-batches-bin/'
_SOURCE_IMAGE_SHAPE = (3, 32, 32)  # Source image format is [image index, channels, y, x]
_VALIDATION_SIZE = 5000


def convert(directory, validation_size, num_threads=1):  # pylint: disable=unused-argument
    """ Convert CIFAR-10 data to tf.train.Examples """
    # TODO: Implement conversion routine
    pass


def dataset(directory, subset):
    """ Return the cifar-10 dataset """
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        {'image/encoded': tf.FixedLenFeature([], tf.string),
         'image/format': tf.FixedLenFeature([], tf.string),
         'image/class/label': tf.FixedLenFeature([1], tf.int64, dtype=tf.int64)},
        {'image': slim.tfexample_decoder.Image(shape=IMAGE_SHAPE, channels=1),
         'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    )

    filenames = encode_utils.get_filenames(directory, subset)

    return slim.dataset.Dataset(
        filenames, tf.TFRecordReader, decoder,
        encode_utils.num_examples(filenames), _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)


def _extract_labels_and_images(archive, filenames):
    """ Extract the labels and images of the given filenames from the archive """
    filenames = list(filenames)
    count = _IMAGES_PER_FILE * len(filenames)

    images = np.zeros((count,) + _SOURCE_IMAGE_SHAPE)
    labels = np.zeros(count)

    for index, filename in enumerate(filenames):
        filepath = os.path.join(_DATA_DIRECTORY, filename)
        print('Extracting', filepath)
        bytestream = archive.extractfile(filepath)

        # Read labels and images
        record_count = _IMAGES_PER_FILE * (np.prod(_SOURCE_IMAGE_SHAPE) + 1)
        buf = bytestream.read(np.sum(record_count))

        dtype = [('labels', np.uint8), ('images', np.uint8, _SOURCE_IMAGE_SHAPE)]
        data = np.frombuffer(buf, dtype=dtype)
        data = data.view(np.recarray)

        start = index * _IMAGES_PER_FILE
        end = start + _IMAGES_PER_FILE
        labels[start:end] = data.labels
        images[start:end] = data.images

    # Convert from source image format to [image index, y, x, channels] format
    images = np.transpose(images, [0, 2, 3, 1])
    images = images.astype(np.float32)
    images = images / 255.0
    return images, labels


def dataset_preloaded(directory, subset):
    """ Return the cifar-10 dataset """
    local_file = learn.datasets.base.maybe_download(_DATAFILE, directory, _DOWNLOAD_URL + _DATAFILE)

    if subset == 'test':
        data_files = ('test_batch.bin',)
    else:
        data_files = ('data_batch_{0}.bin'.format(i + 1) for i in xrange(0, 5))

    print('Opening', local_file)
    archive = tarfile.open(local_file, 'r:gz')
    images, labels = _extract_labels_and_images(archive, data_files)

    if subset == 'train':
        images = images[:-_VALIDATION_SIZE]
        labels = labels[:-_VALIDATION_SIZE]
    elif subset == 'validate':
        images = images[-_VALIDATION_SIZE:]
        labels = labels[-_VALIDATION_SIZE:]

    return slim.dataset.Dataset(
        (images, labels), None, None, images.shape[0], _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)


def _main(argv=None):  # pylint: disable=unused-argument
    """ The main entry point """
    convert(FLAGS.directory, FLAGS.validation_size, FLAGS.num_threads)


def _parse_commandline():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--directory',
        type=str,
        default='data/cifar10',
        action='store',
        help='The directory where to store the CIFAR-10 data'
    )
    parser.add_argument(
        '--num_threads',
        type=str,
        default=1,
        action='store',
        help='How many threads to use to process the images'
    )
    parser.add_argument(
        '--validation_size',
        type=str,
        default=5000,
        action='store',
        help='How many training examples to reserve for validation'
    )

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = _parse_commandline()
    tf.app.run(main=_main)
