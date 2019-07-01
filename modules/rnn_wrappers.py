# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """


import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from collections import namedtuple
from functools import reduce
from abc import abstractmethod
from typing import Tuple
from tacotron2.tacotron.modules import PreNet


class TransparentRNNCellLike:
    '''
    RNNCell-like base class that do not create scopes
    '''

    @property
    @abstractmethod
    def state_size(self):
        pass

    @property
    @abstractmethod
    def output_size(self):
        pass

    @abstractmethod
    def zero_state(self, batch_size, dtype):
        pass

    @abstractmethod
    def __call__(self, inputs, state):
        pass


class RNNStateHistoryWrapperState(
    namedtuple("RNNStateHistoryWrapperState", ["rnn_state", "rnn_state_history", "time"])):
    pass


class RNNStateHistoryWrapper(TransparentRNNCellLike):

    def __init__(self, cell: RNNCell, max_iter):
        self._cell = cell
        self._max_iter = max_iter

    @property
    def state_size(self):
        return RNNStateHistoryWrapperState(self._cell.state_size,
                                           tf.TensorShape([None, None, self.output_size]),
                                           tf.TensorShape([]))

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        rnn_state = self._cell.zero_state(batch_size, dtype)
        history = tf.zeros(shape=[batch_size, 0, self.output_size], dtype=dtype)
        # avoid Tensor#set_shape which merge unknown shape with known shape
        history._shape_val = tf.TensorShape([None, None, self.output_size])  # pylint: disable=protected-access
        time = tf.zeros([], dtype=tf.int32)
        return RNNStateHistoryWrapperState(rnn_state, history, time)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state: RNNStateHistoryWrapperState):
        output, new_rnn_state = self._cell(inputs, state.rnn_state)
        new_history = tf.concat([state.rnn_state_history,
                                 tf.expand_dims(output, axis=1)], axis=1)
        new_history.set_shape([None, None, self.output_size])
        new_state = RNNStateHistoryWrapperState(new_rnn_state, new_history, state.time + 1)
        return output, new_state


class TransformerWrapperState(namedtuple("TransformerWrapperState", ["rnn_state", "alignments"])):
    pass


class TransformerWrapper(TransparentRNNCellLike):

    def __init__(self, cell: RNNStateHistoryWrapper, transformers, memory_sequence_length):
        self._cell = cell
        self._transformers = transformers
        self._memory_sequence_length = memory_sequence_length

    @property
    def state_size(self):
        return TransformerWrapperState(self._cell.state_size, [(None, None) for _ in self._transformers])

    @property
    def output_size(self):
        return TransformerWrapperState(self._cell.output_size, [(None, None) for _ in self._transformers])

    def zero_state(self, batch_size, dtype):
        def initial_alignment(num_heads):
            ia = tf.zeros([batch_size, 0, 0], dtype)
            ia._shape_val = tf.TensorShape([None, None, None])  # pylint: disable=protected-access
            return [ia] * num_heads

        return TransformerWrapperState(self._cell.zero_state(batch_size, dtype),
                                       [ia for ia in initial_alignment(2) for _ in self._transformers])

    def __call__(self, inputs, state: TransformerWrapperState):
        output, new_rnn_state = self._cell(inputs, state.rnn_state)
        history = new_rnn_state.rnn_state_history

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=self._memory_sequence_length)
            return output, alignments + alignment

        transformed, alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                         self._transformers,
                                         (history, []))
        output_element = transformed[:, -1, :]
        new_state = TransformerWrapperState(new_rnn_state, alignments)
        return output_element, new_state


class OutputMgcLf0AndStopTokenWrapper(RNNCell):

    def __init__(self, cell, mgc_out_units, lf0_out_units, dtype=None):
        super(OutputMgcLf0AndStopTokenWrapper, self).__init__()
        self._mgc_out_units = mgc_out_units
        self._lf0_out_units = lf0_out_units
        self._cell = cell
        self.mgc_out_projection1 = tf.layers.Dense(cell.output_size, activation=tf.nn.tanh, dtype=dtype)
        self.mgc_out_projection2 = tf.layers.Dense(mgc_out_units, dtype=dtype)
        self.lf0_out_projection = tf.layers.Dense(lf0_out_units, dtype=dtype)
        self.stop_token_projection = tf.layers.Dense(1, dtype=dtype)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._mgc_out_units, self._lf0_out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        mgc_output = self.mgc_out_projection2(self.mgc_out_projection1(output))
        lf0_output = self.lf0_out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mgc_output, lf0_output, stop_token), res_state


class DecoderMgcLf0PreNetWrapper(RNNCell):

    def __init__(self, cell: RNNCell, mgc_prenets: Tuple[PreNet], lf0_prenets: Tuple[PreNet]):
        super(DecoderMgcLf0PreNetWrapper, self).__init__()
        self._cell = cell
        self.mgc_prenets = mgc_prenets
        self.lf0_prenets = lf0_prenets

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        mgc_input, lf0_input = inputs
        mgc_prenet_output = reduce(lambda acc, pn: pn(acc), self.mgc_prenets, mgc_input)
        lf0_prenet_output = reduce(lambda acc, pn: pn(acc), self.lf0_prenets, lf0_input)
        prenet_output = tf.concat([mgc_prenet_output, lf0_prenet_output], axis=-1)
        return self._cell(prenet_output, state)


class OutputAndStopTokenTransparentWrapper(TransparentRNNCellLike):

    def __init__(self, cell, out_units, out_projection, stop_token_projection):
        self._out_units = out_units
        self._cell = cell
        self.out_projection = out_projection
        self.stop_token_projection = stop_token_projection

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        mel_output = self.out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mel_output, stop_token), res_state


class OutputMgcLf0AndStopTokenTransparentWrapper(TransparentRNNCellLike):

    def __init__(self, cell, mgc_out_units, lf0_out_units, mgc_out_projection, lf0_out_projection, stop_token_projection):
        self._mgc_out_units = mgc_out_units
        self._lf0_out_units = lf0_out_units
        self._cell = cell
        self.mgc_out_projection = mgc_out_projection
        self.lf0_out_projection = lf0_out_projection
        self.stop_token_projection = stop_token_projection

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return (self._mgc_out_units, self._lf0_out_units, 1)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def __call__(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        mgc_output = self.mgc_out_projection(output)
        lf0_output = self.lf0_out_projection(output)
        stop_token = self.stop_token_projection(output)
        return (mgc_output, lf0_output, stop_token), res_state
