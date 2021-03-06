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

""" Read input from tf.train.Example files """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ttq.data import mnist
from ttq.data import cifar10


DATASET_FUNCS = {
    'mnist': mnist.dataset,
    'mnist_preloaded': mnist.dataset_preloaded,
    'cifar10_preloaded': cifar10.dataset_preloaded,
}
DATASETS = DATASET_FUNCS.keys()


def get_dataset(directory, dataset_name, subset_name):
    """ Get the subset of the passed in dataset from the directory indicated """
    return DATASET_FUNCS[dataset_name](directory, subset_name)


DATASET_SHAPE = {
    'mnist': mnist.IMAGE_SHAPE,
    'mnist_preloaded': mnist.IMAGE_SHAPE,
    'cifar10_preloaded': cifar10.IMAGE_SHAPE,
}


def get_data_shape(dataset_name):
    """ Get the shape of the dataset """
    return DATASET_SHAPE[dataset_name]


def _get_data(dataset, batch_size=None, num_epochs=None, num_readers=1):
    """ Get the subset of the passed in dataset from the directory indicated """
    if batch_size is None:
        raise ValueError('batch_size must not specified')

    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, num_readers=num_readers, num_epochs=num_epochs,
        common_queue_capacity=20 * batch_size, common_queue_min=10 * batch_size)

    [image, label] = data_provider.get(['image', 'label'])
    image = preprocess_image(image)

    return tf.no_op(), {}, image, label


# pylint: disable=unused-argument
def _get_data_preloaded(dataset, num_epochs=None, **kwargs):
    """ Get the subset of the passed in dataset from the directory indicated """
    images, labels = dataset.data_sources

    images_initializer = tf.placeholder(dtype=images.dtype, shape=images.shape)
    images_variable = tf.Variable(images_initializer, trainable=False, collections=[])

    labels_initializer = tf.placeholder(dtype=tf.int64, shape=labels.shape)
    labels_variable = tf.Variable(labels_initializer, trainable=False, collections=[])

    [image, label] = tf.train.slice_input_producer(
        [images_variable, labels_variable], num_epochs=num_epochs)

    init_op = tf.group(images_variable.initializer, labels_variable.initializer)
    init_feed_dict = {images_initializer: images, labels_initializer: labels}

    return init_op, init_feed_dict, preprocess_image(image), label
# pylint: enable=unused-argument


DATASET_PROVIDER_FUNCS = {
    'mnist': _get_data,
    'mnist_preloaded': _get_data_preloaded,
    'cifar10_preloaded': _get_data_preloaded,
}


def get_data(dataset_name, dataset, batch_size, num_epochs=None, num_readers=1):
    """ Get the subset of the passed in dataset from the directory indicated """
    return DATASET_PROVIDER_FUNCS[dataset_name](
        dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_readers=num_readers)


def preprocess_image(image):
    """ Process the passed in image """
    # Ensure image is in the range [0.0, 1.0]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image
