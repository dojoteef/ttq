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

""" The model definition """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.layers as layers

import ttq.utils.ops as ops


def basic_residual(inputs, channels, stride, scope=None):
    """ The compute the residual of the inputs """
    with tf.variable_scope(scope, 'residual', [inputs]):
        outputs = ops.convolution2d_padded(inputs, channels, 3, stride, scope='conv1')

        return layers.convolution2d(
            outputs, channels, (1, 1),
            stride=1, activation_fn=None, normalizer_fn=None, scope='conv2')


def bottleneck_residual(inputs, channels, stride, scope=None):
    """ The compute the residual of the inputs """
    with tf.variable_scope(scope, 'residual', [inputs]):
        outputs = layers.convolution2d(inputs, channels / 4, (1, 1), stride=1, scope='conv1')
        outputs = ops.convolution2d_padded(outputs, channels / 4, 3, stride, scope='conv2')

        return layers.convolution2d(
            outputs, channels, (1, 1),
            stride=1, activation_fn=None, normalizer_fn=None, scope='conv3')


@framework.add_arg_scope
def block(inputs, channels, stride, residual_fn=basic_residual, scope=None):
    """ Combines the shortcut with the residual """
    with tf.variable_scope(scope, 'block', [inputs]):
        input_channels = layers.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preactivation = layers.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preactivation')

        if channels == input_channels:
            shortcut = inputs if stride == 1 else layers.max_pool2d(
                inputs, (1, 1), stride=stride, scope='shortcut')
        else:
            shortcut = layers.convolution2d(
                preactivation, channels, (1, 1),
                stride=stride, activation_fn=None, normalizer_fn=None, scope='shortcut')

        if residual_fn == bottleneck_residual and channels % 4 != 0:
            raise ValueError('The channels for the block must be evenly divisible by 4!')

        outputs = shortcut + residual_fn(preactivation, channels, stride, scope='residual')
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs)

        return outputs


@framework.add_arg_scope
def stack(inputs, channels, count, residual_fn=basic_residual, has_next=True, scope=None):
    """ Create a stack of resnet blocks with the given inputs and configuration """
    with tf.variable_scope(scope, 'block', [inputs]):
        outputs = inputs

        with framework.arg_scope([block], residual_fn=residual_fn):
            for index in xrange(count - 1):
                outputs = block(outputs, channels, 1, scope='element{0}'.format(index))

            # If there are more blocks after the current block then upsample
            stride = 2 if has_next else 1
            return block(outputs, channels, stride, scope='element{0}'.format(count - 1))


@contextlib.contextmanager
def model_arg_scope(config, is_training):
    """ Define the arg scope of the model """
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'is_training': is_training,
        'scale': True,
    }

    with framework.arg_scope(
        [layers.convolution2d],
        biases_initializer=None,
        weights_regularizer=layers.l2_regularizer(config.regularization_scale),
        weights_initializer=layers.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):
        with framework.arg_scope([layers.batch_norm], **batch_norm_params):
            with framework.arg_scope([layers.max_pool2d], padding='SAME') as returned_scope:
                yield returned_scope


def quantize_filter(name):
    """ Given a variable name determine if it should be quantized """
    # Only quantize weights, except in the first or last layer of the model
    return 'weights' not in name or 'conv0' in name or 'logits' in name


def create_model(config, inputs, dataset, is_training=False):
    """ Create the model """
    with tf.variable_scope('resnet', values=[inputs]):
        with model_arg_scope(config, is_training):
            # Let the first residual do the batch norm and activation, so skip them here
            with framework.arg_scope(
                [layers.convolution2d],
                activation_fn=None, normalizer_fn=None):
                model = ops.convolution2d_padded(inputs, 64, 7, stride=2, scope='conv0')
                model = layers.max_pool2d(model, (3, 3), stride=2, scope='pool0')

            if config.bottleneck:
                residual_fn = bottleneck_residual
                block_config = [256, 512, 1024, 2048]
            else:
                residual_fn = basic_residual
                block_config = [64, 128, 256, 512]

            block_scopes = ['block{0}'.format(i) for i in xrange(4)]
            block_sizes = [getattr(config, 'block{0}_size'.format(i + 1)) for i in xrange(4)]
            block_definitions = zip(block_config, block_sizes, block_scopes)

            with framework.arg_scope([stack], residual_fn=residual_fn):
                for index, definition in enumerate(block_definitions):
                    channels, size, name = definition
                    has_next = index < len(block_definitions) - 1
                    model = stack(model, channels, size, scope=name, has_next=has_next)

            model = layers.batch_norm(model, activation_fn=tf.nn.relu, scope='norm')
            model = tf.reduce_mean(model, [1, 2], name='global_avg_pool', keep_dims=True)

            outputs = layers.convolution2d(
                model, dataset.num_classes, [1, 1],
                activation_fn=None, normalizer_fn=None, scope='logits')

            return tf.squeeze(outputs, [1, 2])
