# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism


class ScaledDotProductAttentionMechanism(AttentionMechanism):
    def __init__(self, keys, values, num_heads, drop_rate=0.0, is_training=False, use_padding_mask=True,
                 use_subsequent_mask=False):
        self._keys = keys  # (B, num_heads, T, C)
        self._values = values
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.use_padding_mask = use_padding_mask
        self.use_subsequent_mask = use_subsequent_mask
        self.is_training = is_training

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def alignments_size(self):
        return [self.num_heads, self.keys.shape[-2].value] or tf.shape(self.keys)[1:3]

    @property
    def state_size(self):
        return self.alignments_size

    def initial_alignment(self, batch_size, dtype):
        key_length = self.alignments_size
        size = tf.stack([batch_size, tf.ones(shape=(), dtype=tf.int32), key_length], axis=0)
        return tf.zeros(size, dtype=dtype)

    def __call__(self, query, memory_sequence_length=None):
        # Q K^\top
        x = tf.matmul(query, self.keys, transpose_b=True)

        # scale attention output
        s = tf.cast(tf.shape(query)[-1], dtype=query.dtype)
        x = x / tf.sqrt(s)

        x = self.apply_padding_mask(x, memory_sequence_length) if self.use_padding_mask else x
        x = self.apply_subsequent_mask(x) if self.use_subsequent_mask else x

        # softmax over last dim
        # (B, num_heads, T_query, T_memory)
        x = tf.nn.softmax(x, axis=-1)
        alignment_scores = x

        x = tf.layers.dropout(x, rate=self.drop_rate, training=self.is_training)

        x = tf.matmul(x, self.values)

        return x, alignment_scores

    def apply_padding_mask(self, score, memory_sequence_length, score_mask_value=-np.inf):
        max_length = tf.shape(self.keys)[2]
        score_mask = tf.sequence_mask(
            memory_sequence_length, maxlen=max_length)
        # (B, T) -> (B, 1, T)
        score_mask = tf.expand_dims(score_mask, axis=1)
        # (B, 1, T) -> (B, T, T)
        score_mask = score_mask & tf.transpose(score_mask, perm=[0, 2, 1])
        # (B, 1, T, T) -> (B, num_heads, T, T)
        score_mask = tf.stack([score_mask] * self.num_heads, axis=1)
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)

    def apply_subsequent_mask(self, score, score_mask_value=-np.inf):
        batch_size = tf.shape(self.keys)[0]
        max_length = tf.shape(self.keys)[2]
        score_mask = tf.ones([batch_size, self.num_heads, 1, 1], dtype=tf.bool) & tf.matrix_band_part(
            tf.ones([max_length, max_length], dtype=tf.bool), -1, 0)
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)


class MultiHeadAttention(tf.layers.Layer):

    def __init__(self, model_dim, num_heads, drop_rate, is_training,
                 use_padding_mask=False, use_subsequent_mask=False,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MultiHeadAttention, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.drop_rate = drop_rate
        self.is_training = is_training
        self.use_padding_mask = use_padding_mask
        self.use_subsequent_mask = use_subsequent_mask
        # ToDo: remove bias from projections
        self.key_projection = tf.layers.Dense(model_dim, dtype=dtype)
        self.value_projection = tf.layers.Dense(model_dim, dtype=dtype)
        self.query_projection = tf.layers.Dense(model_dim, dtype=dtype)
        self.output_projection = tf.layers.Dense(model_dim, dtype=dtype)

    def call(self, inputs, memory_sequence_length=None):
        key, value, query = inputs
        shape = tf.shape(key)
        head_shape = [shape[0], shape[1], self.num_heads, self.head_dim]
        # (B, T, model_dim) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        key_projected = tf.transpose(tf.reshape(self.key_projection(key), shape=head_shape), perm=[0, 2, 1, 3])
        key_projected.set_shape([None, self.num_heads, None, self.head_dim])
        value_projected = tf.transpose(tf.reshape(self.value_projection(value), shape=head_shape), perm=[0, 2, 1, 3])
        value_projected.set_shape([None, self.num_heads, None, self.head_dim])
        query_projected = tf.transpose(tf.reshape(self.query_projection(query), shape=head_shape), perm=[0, 2, 1, 3])
        query_projected.set_shape([None, self.num_heads, None, self.head_dim])
        attention_mechanism = ScaledDotProductAttentionMechanism(key_projected, value_projected, self.num_heads,
                                                                 drop_rate=self.drop_rate,
                                                                 is_training=self.is_training,
                                                                 use_padding_mask=self.use_padding_mask,
                                                                 use_subsequent_mask=self.use_subsequent_mask)
        x, alignment = attention_mechanism(query_projected, memory_sequence_length=memory_sequence_length)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), shape=[shape[0], shape[1], self.num_heads * self.head_dim])
        output = self.output_projection(x)
        alignment = [alignment[:, i, :, :] for i in range(self.num_heads)]
        return output, alignment


class SelfAttention(tf.layers.Layer):

    def __init__(self, model_dim, num_heads, drop_rate, is_training,
                 use_padding_mask=False, use_subsequent_mask=False,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SelfAttention, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.attention = MultiHeadAttention(model_dim, num_heads, drop_rate, is_training,
                                            use_padding_mask=use_padding_mask,
                                            use_subsequent_mask=use_subsequent_mask,
                                            dtype=dtype)

    def call(self, inputs, memory_sequence_length=None):
        key, value, query = (inputs, inputs, inputs)
        return self.attention((key, value, query), memory_sequence_length=memory_sequence_length)
