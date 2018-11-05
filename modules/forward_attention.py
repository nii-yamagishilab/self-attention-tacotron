# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention
from collections import namedtuple


def _location_sensitive_score(W_query, W_fill, W_keys):
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or tf.shape(W_keys)[-1]

    v_a = tf.get_variable("attention_variable",
                          shape=[num_units],
                          dtype=dtype,
                          initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable("attention_bias",
                          shape=[num_units],
                          dtype=dtype,
                          initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fill + b_a), axis=[2])


def _calculate_context(alignments, values):
    '''
    This is a duplication of tensorflow.contrib.seq2seq.attention_wrapper._compute_attention.
    ToDo: Avoid the redundant computation. This requires abstraction of AttentionWrapper itself.
    :param alignments: [batch_size, 1, memory_time]
    :param values: [batch_size, memory_time, memory_size]
    :return:
    '''
    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    context = tf.matmul(expanded_alignments, values)  # [batch_size, 1, memory_size]
    context = tf.squeeze(context, [1])  # [batch_size, memory_size]
    return context


class ForwardAttentionState(namedtuple("ForwardAttentionState", ["alignments", "alpha", "u"])):
    pass


class ForwardAttention(BahdanauAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length,
                 attention_kernel,
                 attention_filters,
                 use_transition_agent=False,
                 cumulative_weights=True,
                 name="ForwardAttention"):
        super(ForwardAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            probability_fn=None,
            name=name)
        self._use_transition_agent = use_transition_agent
        self._cumulative_weights = cumulative_weights

        self.location_convolution = tf.layers.Conv1D(filters=attention_filters,
                                                     kernel_size=attention_kernel,
                                                     padding="SAME",
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer(),
                                                     name="location_features_convolution")

        self.location_layer = tf.layers.Dense(units=num_units,
                                              use_bias=False,
                                              dtype=memory.dtype,
                                              name="location_features_layer")

        if use_transition_agent:
            # ToDo: support speed control bias
            self.transition_factor_projection = tf.layers.Dense(units=1,
                                                                use_bias=True,
                                                                dtype=memory.dtype,
                                                                activation=tf.nn.sigmoid,
                                                                name="transition_factor_projection")

    def __call__(self, query, state):
        previous_alignments, prev_alpha, prev_u = state
        with tf.variable_scope(None, "location_sensitive_attention", [query]):
            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query

            # -> [batch_size, 1, attention_dim]
            expanded_processed_query = tf.expand_dims(processed_query, 1)

            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            processed_location_features = self.location_layer(f)

            energy = _location_sensitive_score(expanded_processed_query, processed_location_features, self.keys)

        alignments = self._probability_fn(energy, state)

        # forward attention
        prev_alpha_n_minus_1 = tf.pad(prev_alpha[:, :-1], paddings=[[0, 0], [1, 0]])
        alpha = ((1 - prev_u) * prev_alpha + prev_u * prev_alpha_n_minus_1 + 1e-7) * alignments
        alpha_normalized = alpha / tf.reduce_sum(alpha, axis=1, keep_dims=True)
        if self._use_transition_agent:
            context = _calculate_context(alpha_normalized, self.values)
            transition_factor_input = tf.concat([context, processed_query], axis=-1)
            transition_factor = self.transition_factor_projection(transition_factor_input)
        else:
            transition_factor = prev_u

        if self._cumulative_weights:
            next_state = ForwardAttentionState(alignments + previous_alignments, alpha_normalized, transition_factor)
        else:
            next_state = ForwardAttentionState(alignments, alpha_normalized, transition_factor)
        return alpha_normalized, next_state

    @property
    def state_size(self):
        return ForwardAttentionState(self._alignments_size, self._alignments_size, 1)

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        # alpha_0 = 1, alpha_n = 0 where n = 2, 3, ..., N
        initial_alpha = tf.concat([
            tf.ones([batch_size, 1], dtype=dtype),
            tf.zeros_like(initial_alignments, dtype=dtype)[:, 1:]], axis=1)
        # transition factor
        initial_u = 0.5 * tf.ones([batch_size, 1], dtype=dtype)
        return ForwardAttentionState(initial_alignments, initial_alpha, initial_u)
