# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """


import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention


class TeacherForcingForwardAttention(BahdanauAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length,
                 teacher_alignments,
                 name="ForwardAttention"):
        super(TeacherForcingForwardAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            probability_fn=None,
            name=name)
        self.teacher_alignments = teacher_alignments

    def __call__(self, query, state):
        previous_alignments, prev_index = state

        index = prev_index + 1
        alignments = self.teacher_alignments[:, index]
        next_state = (alignments, index)
        return alignments, next_state

    @property
    def state_size(self):
        return self._alignments_size, 1

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        initial_index = tf.to_int64(-1)
        return initial_alignments, initial_index


class TeacherForcingAdditiveAttention(BahdanauAttention):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length,
                 teacher_alignments,
                 name="BahdanauAttention"):
        super(TeacherForcingAdditiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            probability_fn=None,
            name=name)
        self.teacher_alignments = teacher_alignments

    def __call__(self, query, state):
        previous_alignments, prev_index = state

        index = prev_index + 1
        alignments = self.teacher_alignments[:, index]
        next_state = (alignments, index)
        return alignments, next_state

    @property
    def state_size(self):
        return self._alignments_size, 1

    def initial_state(self, batch_size, dtype):
        initial_alignments = self.initial_alignments(batch_size, dtype)
        initial_index = tf.to_int64(-1)
        return initial_alignments, initial_index
