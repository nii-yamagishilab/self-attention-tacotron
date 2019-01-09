# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from modules.attentions import AttentionOptions, attention_mechanism_factory


def attention_factory(params):
    attention_option = AttentionOptions(attention=params.attention,
                                        num_units=params.attention_out_units,
                                        attention_kernel=params.attention_kernel,
                                        attention_filters=params.attention_filters,
                                        smoothing=False,
                                        cumulative_weights=params.cumulative_weights,
                                        use_transition_agent=params.use_forward_attention_transition_agent)
    return attention_mechanism_factory(attention_option)


def dual_source_attention_factory(params):
    attention_option1 = AttentionOptions(attention=params.attention,
                                         num_units=params.attention1_out_units,
                                         attention_kernel=params.attention_kernel,
                                         attention_filters=params.attention_filters,
                                         smoothing=False,
                                         cumulative_weights=params.cumulative_weights,
                                         use_transition_agent=params.use_forward_attention_transition_agent)
    attention_option2 = AttentionOptions(attention=params.attention2,
                                         num_units=params.attention2_out_units,
                                         attention_kernel=params.attention_kernel,
                                         attention_filters=params.attention_filters,
                                         smoothing=False,
                                         cumulative_weights=params.cumulative_weights,
                                         use_transition_agent=params.use_forward_attention_transition_agent)
    return attention_mechanism_factory(attention_option1), attention_mechanism_factory(attention_option2)


def force_alignment_attention_factory(params):
    attention_option = AttentionOptions(attention=params.forced_alignment_attention,
                                        num_units=params.attention_out_units,
                                        attention_kernel=params.attention_kernel,
                                        attention_filters=params.attention_filters,
                                        smoothing=False,
                                        cumulative_weights=params.cumulative_weights,
                                        use_transition_agent=params.use_forward_attention_transition_agent)
    return attention_mechanism_factory(attention_option)


def force_alignment_dual_source_attention_factory(params):
    attention_option1 = AttentionOptions(attention=params.forced_alignment_attention,
                                         num_units=params.attention1_out_units,
                                         attention_kernel=params.attention_kernel,
                                         attention_filters=params.attention_filters,
                                         smoothing=False,
                                         cumulative_weights=params.cumulative_weights,
                                         use_transition_agent=params.use_forward_attention_transition_agent)
    attention_option2 = AttentionOptions(attention=params.forced_alignment_attention2,
                                         num_units=params.attention2_out_units,
                                         attention_kernel=params.attention_kernel,
                                         attention_filters=params.attention_filters,
                                         smoothing=False,
                                         cumulative_weights=params.cumulative_weights,
                                         use_transition_agent=params.use_forward_attention_transition_agent)
    return attention_mechanism_factory(attention_option1), attention_mechanism_factory(attention_option2)
