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

""" Additional utility operations """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.layers as layers


def reduce_prod(iterable):
    """ Return the product of the iterable """
    return reduce(lambda x, y: x * y, iterable)


def exponential_decay(batch_size, num_epochs, initial_rate, decay_rate, dataset,
                      staircase=True, name=None):
    """ Get the exponential decay for the following parameters """
    global_step = framework.get_or_create_global_step()
    decay_steps = int(num_epochs * dataset.num_samples / batch_size)

    return tf.train.exponential_decay(
        initial_rate, global_step,
        decay_steps, decay_rate,
        staircase=staircase, name=name)


def convolution2d_padded(inputs, num_outputs, kernel_size, stride, scope=None):
    """ Convolution using 'SAME' padding if stride is 1; zero padding with 'VALID' otherwise """
    if stride == 1:
        return layers.convolution2d(inputs, num_outputs, kernel_size, scope=scope)

    padding_size = kernel_size - 1
    padding_size = (padding_size // 2, padding_size - (padding_size // 2))
    inputs = tf.pad(inputs, ((0, 0), padding_size, padding_size, (0, 0)))

    return layers.convolution2d(
        inputs, num_outputs, kernel_size,
        stride=stride, padding='VALID', scope=scope)
