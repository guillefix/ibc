# coding=utf-8
# Copyright 2022 The Reach ML Authors.
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

"""Implements a tf_agents compatible mlp-ebm."""
import gin
from ibc.networks.layers import mlp_dropout
from ibc.networks.layers import resnet
from ibc.networks.layers import spectral_norm
import tensorflow as tf
from tf_agents.networks import network

from transformers import BertTokenizer, TFBertModel
import copy
from collections import OrderedDict

from transformers import AutoConfig, TFAutoModel
#config = AutoConfig.from_pretrained('bert-base-uncased', vocab_size=73)
config = AutoConfig.from_pretrained('bert-base-uncased', vocab_size=73, num_hidden_layers=6, num_attention_heads=8, hidden_size=512)


# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
#
# outputs = model(inputs)
# inputs
#
# last_hidden_states = outputs.last_hidden_state
# last_hidden_states.shape

@gin.configurable
class MLPEBMLang(network.Network):
  """MLP-EBM compatible with tfagents."""

  def __init__(self,
               obs_spec,
               action_spec,
               width=512,
               depth=2,
               rate=0.1,
               name='MLPEBM',
               activation='relu',
               layers='MLPDropout',
               kernel_initializer='normal',
               bias_initializer='normal',
               dense_layer_type='regular'):

    # import pdb; pdb.set_trace()
    # obs_spec[0]['annotation'] = tf.TensorSpec(shape=(1,784), dtype=tf.float32, name='observation/annotation')
    super(MLPEBMLang, self).__init__(
        input_tensor_spec=obs_spec, state_spec=(), name=name)

    if dense_layer_type == 'regular':
      dense = tf.keras.layers.Dense
    elif dense_layer_type == 'spectral_norm':
      dense = spectral_norm.DenseSN
    else:
      raise ValueError('Unexpected dense layer type')

    # Define MLP.
    hidden_sizes = [width for _ in range(depth)]
    if layers == 'MLPDropout':
      self._mlp = mlp_dropout.MLPDropoutLayer(
          hidden_sizes, rate, kernel_initializer, bias_initializer,
          dense, activation)
    elif layers == 'ResNetOrig':
      self._mlp = resnet.ResNetOrigLayer(
          hidden_sizes, rate, kernel_initializer, bias_initializer,
          dense, activation)
    elif layers == 'ResNetPreActivation':
      self._mlp = resnet.ResNetPreActivationLayer(
          hidden_sizes, rate, kernel_initializer, bias_initializer,
          dense, activation)

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #model = TFBertModel.from_pretrained("bert-base-uncased")
    model =  TFAutoModel.from_config(config)
    # self._tokenizer = tokenizer
    self._encoder = model
    #TODO: randomly reinitialize?

    # Define projection to energy.
    self._project_energy = dense(
        action_spec.shape[-1],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

    self._cached_inputs = None
    self._cached_latents = None


  def call(self, inputs, training, step_type=(), network_state=()):
    # obs: dict of named obs_spec.
    # act:   [B x act_spec]
    obs, act = inputs
    obs = tf.nest.map_structure(tf.identity, obs)
    obs = dict(obs)
    # obs = copy.deepcopy(dict(obs))

    # print(obs)
    # import pdb; pdb.set_trace()
    inputs = obs['annotation'][:,0,:]
    print(inputs == self._cached_inputs)
    print(inputs)
    print(self._cached_inputs)
    if self._cached_latents is not None and self._cached_inputs is not None and inputs == self._cached_inputs:
        print("found latents in cache")
        obs['annotation'] = self._cached_latents
    else:
        self._cached_inputs = inputs
        latents = self._encoder(inputs).last_hidden_state
        # import pdb; pdb.set_trace()
        obs['annotation'] = latents[:,-1:,:]
        self._cached_latents = obs['annotation']

    # Combine dict of observations to concatenated tensor. [B x T x obs_spec]
    # import pdb; pdb.set_trace()
    obs = tf.concat(tf.nest.flatten(obs), axis=-1)

    # Flatten obs across time: [B x T * obs_spec]
    batch_size = tf.shape(obs)[0]
    obs = tf.reshape(obs, [batch_size, -1])

    # Concat [obs, act].
    x = tf.concat([obs, act], -1)

    # Forward mlp.
    x = self._mlp(x, training=training)

    # Project to energy.
    x = self._project_energy(x, training=training)

    # import pdb; pdb.set_trace()
    # Squeeze extra dim.
    x = tf.squeeze(x, axis=-1)

    return x, network_state
