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
                                                       "memory",
                                                       "memory_sequence_length",
                                                       "attention_kernel",
                                                       "attention_filters",
                                                       "smoothing",
                                                       "cumulative_weights",
                                                       "use_transition_agent",
                                                       "teacher_alignments"])):
    pass


def attention_mechanism_factory(options: AttentionOptions):
    if options.attention == "forward":
        mechanism = ForwardAttention(options.num_units,
                                     options.memory,
                                     memory_sequence_length=options.memory_sequence_length,
                                     attention_kernel=options.attention_kernel,
                                     attention_filters=options.attention_filters,
                                     use_transition_agent=options.use_transition_agent,
                                     cumulative_weights=options.cumulative_weights)
    elif options.attention == "location_sensitive":
        mechanism = LocationSensitiveAttention(options.num_units, options.memory,
                                               memory_sequence_length=options.memory_sequence_length,
                                               attention_kernel=options.attention_kernel,
                                               attention_filters=options.attention_filters,
                                               smoothing=options.smoothing,
                                               cumulative_weights=options.cumulative_weights)
    elif options.attention == "teacher_forcing_forward":
        mechanism = TeacherForcingForwardAttention(options.num_units,
                                                   options.memory,
                                                   options.memory_sequence_length,
                                                   options.teacher_alignments)
    elif options.attention == "teacher_forcing_additive":
        mechanism = TeacherForcingAdditiveAttention(options.num_units,
                                                    options.memory,
                                                    options.memory_sequence_length,
                                                    options.teacher_alignments)
    elif options.attention == "additive":
        mechanism = BahdanauAttention(options.num_units, options.memory,
                                      memory_sequence_length=options.memory_sequence_length)
    else:
        raise ValueError(f"Unknown attention mechanism: {options.attention}")
    return mechanism
