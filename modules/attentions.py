# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from modules.forward_attention import ForwardAttention
from modules.teacher_forcing_attention import TeacherForcingForwardAttention, TeacherForcingAdditiveAttention
from tacotron2.tacotron.tacotron_v2 import LocationSensitiveAttention
from tensorflow.contrib.seq2seq import BahdanauAttention
from collections import namedtuple


class AttentionOptions(namedtuple("AttentionOptions", ["attention",
                                                       "num_units",
                                                       "attention_kernel",
                                                       "attention_filters",
                                                       "smoothing",
                                                       "cumulative_weights",
                                                       "use_transition_agent"])):
    pass


def attention_mechanism_factory(options: AttentionOptions):
    def attention_fn(memory, memory_sequence_length, teacher_alignments=None):
        if options.attention == "forward":
            mechanism = ForwardAttention(num_units=options.num_units,
                                         memory=memory,
                                         memory_sequence_length=memory_sequence_length,
                                         attention_kernel=options.attention_kernel,
                                         attention_filters=options.attention_filters,
                                         use_transition_agent=options.use_transition_agent,
                                         cumulative_weights=options.cumulative_weights)
        elif options.attention == "location_sensitive":
            mechanism = LocationSensitiveAttention(num_units=options.num_units,
                                                   memory=memory,
                                                   memory_sequence_length=memory_sequence_length,
                                                   attention_kernel=options.attention_kernel,
                                                   attention_filters=options.attention_filters,
                                                   smoothing=options.smoothing,
                                                   cumulative_weights=options.cumulative_weights)
        elif options.attention == "teacher_forcing_forward":
            mechanism = TeacherForcingForwardAttention(num_units=options.num_units,
                                                       memory=memory,
                                                       memory_sequence_length=memory_sequence_length,
                                                       teacher_alignments=teacher_alignments)
        elif options.attention == "teacher_forcing_additive":
            mechanism = TeacherForcingAdditiveAttention(num_units=options.num_units,
                                                        memory=memory,
                                                        memory_sequence_length=memory_sequence_length,
                                                        teacher_alignments=teacher_alignments)
        elif options.attention == "additive":
            mechanism = BahdanauAttention(num_units=options.num_units,
                                          memory=memory,
                                          memory_sequence_length=memory_sequence_length,
                                          dtype=memory.dtype)
        else:
            raise ValueError(f"Unknown attention mechanism: {options.attention}")
        return mechanism

    return attention_fn
