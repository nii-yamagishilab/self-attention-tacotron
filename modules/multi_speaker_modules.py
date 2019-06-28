# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf


class MultiSpeakerPreNet(tf.layers.Layer):

    def __init__(self, out_units, speaker_embed, is_training, drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MultiSpeakerPreNet, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.out_units = out_units
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.dense0 = tf.layers.Dense(out_units, activation=tf.nn.relu, dtype=dtype)
        self.dense = tf.layers.Dense(out_units, activation=tf.nn.relu, dtype=dtype)
        self.speaker_projection = tf.layers.Dense(out_units, activation=tf.nn.softsign, dtype=dtype)
        self.speaker_embed = speaker_embed

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        dense0 = self.dense0(inputs)
        dense0 += self.speaker_projection(self.speaker_embed)
        dense = self.dense(dense0)
        dropout = tf.layers.dropout(dense, rate=self.drop_rate, training=self.is_training)
        return dropout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


