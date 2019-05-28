# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.contrib.seq2seq import Helper


class TransformerTrainingHelper(Helper):

    def __init__(self, targets, output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        t_shape = tf.shape(targets)
        self._batch_size = t_shape[0]
        self._output_dim = output_dim
        self.n_feed_frame = n_feed_frame

        self._targets = tf.reshape(targets,
                                   shape=tf.stack([self.batch_size, t_shape[1] // r, tf.to_int32(output_dim * r)]))
        self._targets.set_shape((targets.get_shape()[0].value, None, output_dim * r))

        # Use full length for every target because we don't want to mask the padding frames
        num_steps = tf.shape(self._targets)[1]
        self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (
            tf.tile([False], [self._batch_size]),
            _go_frames(self._batch_size, self._output_dim * self.n_feed_frame, self._targets.dtype))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        finished = (time + 1 >= self._lengths)
        next_inputs = self._targets[:, time, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([outputs.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


class StopTokenBasedMgcLf0InferenceHelper(Helper):

    def __init__(self, batch_size, mgc_output_dim, lf0_output_dim, r, n_feed_frame=1, min_iters=10, dtype=None):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._mgc_output_dim = mgc_output_dim
        self._lf0_output_dim = lf0_output_dim
        self.r = r
        self.n_feed_frame = n_feed_frame
        self.min_iters = min_iters
        self._dtype = dtype or backend.floatx()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        initial_mgc_lf0 = (_go_frames(self._batch_size, self._mgc_output_dim * self.n_feed_frame, self._dtype),
                           _go_frames(self._batch_size, self._lf0_output_dim * self.n_feed_frame, self._dtype))
        return tf.tile([False], [self._batch_size]), initial_mgc_lf0

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        mgc_output, lf0_output, done = outputs
        finished = self.is_finished(done, time)
        next_mgc_inputs = mgc_output[:, -self._mgc_output_dim * self.n_feed_frame:]
        next_mgc_inputs.set_shape([mgc_output.get_shape()[0].value, self._mgc_output_dim * self.n_feed_frame])
        lf0_output = tf.nn.softmax(tf.reshape(lf0_output, shape=[-1, self.r, self._lf0_output_dim]), axis=-1)
        next_lf0_inputs = tf.reshape(lf0_output[:, -self.n_feed_frame:, :],
                                     shape=[-1, self._lf0_output_dim * self.n_feed_frame])
        next_lf0_inputs.set_shape([lf0_output.get_shape()[0].value, self._lf0_output_dim * self.n_feed_frame])
        next_inputs = (next_mgc_inputs, next_lf0_inputs)
        return (finished, next_inputs, state)

    def is_finished(self, done, time):
        termination_criteria = tf.greater(tf.nn.sigmoid(done), 0.5)
        minimum_requirement = tf.greater(time, self.min_iters)
        termination = tf.logical_and(termination_criteria, minimum_requirement)
        return tf.reduce_all(termination, axis=0)


class ValidationMgcLf0Helper(Helper):

    def __init__(self, mgc_targets, lf0_targets, batch_size, mgc_output_dim, lf0_output_dim, r, n_feed_frame=1,
                 teacher_forcing=False):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._mgc_output_dim = mgc_output_dim
        self._lf0_output_dim = lf0_output_dim
        self._end_token = tf.tile([0.0], [mgc_output_dim * r])
        self.r = r
        self.n_feed_frame = n_feed_frame
        self.num_steps = tf.shape(mgc_targets)[1] // r
        self.teacher_forcing = teacher_forcing
        self._mgc_targets = tf.reshape(mgc_targets,
                                       shape=tf.stack(
                                           [self.batch_size, self.num_steps, tf.to_int32(mgc_output_dim * r)]))
        self._lf0_targets = tf.reshape(lf0_targets,
                                       shape=tf.stack(
                                           [self.batch_size, self.num_steps, tf.to_int32(lf0_output_dim * r)]))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        initial_mgc_lf0 = (_go_frames(self._batch_size, self._mgc_output_dim * self.n_feed_frame,
                                      self._mgc_targets.dtype),
                           _go_frames(self._batch_size, self._lf0_output_dim * self.n_feed_frame,
                                      self._mgc_targets.dtype))
        return tf.tile([False], [self._batch_size]), initial_mgc_lf0

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        mgc_output, lf0_output, done = outputs
        finished = (time + 1 >= self.num_steps)
        next_mgc_inputs = self._mgc_targets[:, time,
                          -self._mgc_output_dim * self.n_feed_frame:] if self.teacher_forcing else mgc_output[:,
                                                                                                   -self._mgc_output_dim * self.n_feed_frame:]
        next_mgc_inputs.set_shape([mgc_output.get_shape()[0].value, self._mgc_output_dim * self.n_feed_frame])
        lf0_output = tf.nn.softmax(tf.reshape(lf0_output, shape=[-1, self.r, self._lf0_output_dim]), axis=-1)
        next_lf0_inputs = self._lf0_targets[:, time,
                          -self._lf0_output_dim * self.n_feed_frame:] if self.teacher_forcing else tf.reshape(
            lf0_output[:, -self.n_feed_frame:, :],
            shape=[-1, self._lf0_output_dim * self.n_feed_frame])
        next_lf0_inputs.set_shape([lf0_output.get_shape()[0].value, self._lf0_output_dim * self.n_feed_frame])
        return (finished, (next_mgc_inputs, next_lf0_inputs), state)


class TrainingMgcLf0Helper(Helper):

    def __init__(self, mgc_targets, lf0_targets, mgc_output_dim, lf0_output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        t_shape = tf.shape(mgc_targets)
        self._batch_size = t_shape[0]
        self._mgc_output_dim = mgc_output_dim
        self._lf0_output_dim = lf0_output_dim
        self.n_feed_frame = n_feed_frame

        self._mgc_targets = tf.reshape(mgc_targets,
                                       shape=tf.stack(
                                           [self.batch_size, t_shape[1] // r, tf.to_int32(mgc_output_dim * r)]))
        self._mgc_targets.set_shape((None, None, mgc_output_dim * r))
        self._lf0_targets = tf.reshape(lf0_targets,
                                       shape=tf.stack(
                                           [self.batch_size, t_shape[1] // r, tf.to_int32(lf0_output_dim * r)]))
        self._lf0_targets.set_shape((None, None, lf0_output_dim * r))

        # Use full length for every target because we don't want to mask the padding frames
        num_steps = tf.shape(self._mgc_targets)[1]
        self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        initial_mgc_lf0 = (_go_frames(self._batch_size, self._mgc_output_dim * self.n_feed_frame,
                                      self._mgc_targets.dtype),
                           _go_frames(self._batch_size, self._lf0_output_dim * self.n_feed_frame,
                                      self._mgc_targets.dtype))
        return tf.tile([False], [self._batch_size]), initial_mgc_lf0

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        finished = (time + 1 >= self._lengths)
        next_mgc_inputs = self._mgc_targets[:, time, -self._mgc_output_dim * self.n_feed_frame:]
        next_mgc_inputs.set_shape([None, self._mgc_output_dim * self.n_feed_frame])
        next_lf0_inputs = self._lf0_targets[:, time, -self._lf0_output_dim * self.n_feed_frame:]
        next_lf0_inputs.set_shape([None, self._lf0_output_dim * self.n_feed_frame])
        return (finished, (next_mgc_inputs, next_lf0_inputs), state)


def _go_frames(batch_size, output_dim, dtype):
    return tf.tile(tf.convert_to_tensor([[0.0]], dtype=dtype), [batch_size, output_dim])
