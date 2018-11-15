# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf

hparams = tf.contrib.training.HParams(

    # Audio
    num_mels=80,
    num_mgcs=60,
    num_freq=2049,
    sample_rate=48000,
    frame_length_ms=50.0,
    frame_shift_ms=12.5,
    ref_level_db=20,
    average_mel_level_db=[0.0],
    stddev_mel_level_db=[0.0],
    silence_mel_level_db=-3.0,

    ## MGC
    mgc_dim=60,
    mgc_alpha=0.77,
    mgc_gamma=0.0,
    mgc_fft_len=4096,

    ## LF0
    num_lf0s=256,
    f0_max=529.0,
    f0_min=66.0,
    lf0_loss_factor=0.5,

    # Dataset
    dataset="vctk.dataset.DatasetSource",  # vctk.dataset.DatasetSource
    num_symbols=256,
    source_file_extension="source.tfrecord",
    target_file_extension="target.tfrecord",

    # Model:
    # tacotron_model= ExtendedTacotronV1Model, DualSourceSelfAttentionTacotronModel, DualSourceSelfAttentionMgcLf0TacotronModel
    tacotron_model="ExtendedTacotronV1Model",
    outputs_per_step=2,
    n_feed_frame=2,

    ## Embedding
    embedding_dim=256,

    ### accent
    use_accent_type=False,
    accent_type_embedding_dim=32,
    num_accent_type=129,
    accent_type_offset=0x3100,
    accent_type_unknown=0x3180,
    accent_type_prenet_out_units=(32, 16),
    encoder_prenet_out_units_if_accent=(224, 112),

    ## Encoder
    # encoder= ZoneoutEncoderV1 | EncoderV1WithAccentType | SelfAttentionCBHGEncoder | SelfAttentionCBHGEncoderWithAccentType
    encoder="ZoneoutEncoderV1",

    ### Encoder V1
    encoder_prenet_drop_rate=0.5,
    cbhg_out_units=256,
    conv_channels=128,
    max_filter_width=16,
    projection1_out_channels=128,
    projection2_out_channels=128,
    num_highway=4,
    encoder_prenet_out_units=(256, 128),

    ### Self Attention
    self_attention_out_units=32,
    self_attention_num_heads=2,
    self_attention_num_hop=1,
    self_attention_encoder_out_units=32,
    self_attention_drop_rate=0.05,
    self_attention_transformer_num_conv_layers=1,
    self_attention_transformer_kernel_size=5,

    ## Decoder
    decoder="ExtendedDecoder", # ExtendedDecoder | TransformerDecoder | DualSourceDecoder | DualSourceTransformerDecoder | MgcLf0DualSourceDecoder
    attention="additive",  # additive, location_sensitive, forward
    forced_alignment_attention = "teacher_forcing_forward",  # teacher_forcing_forward, teacher_forcing_additive

    ### Dual Source Decoder
    attention2="additive",
    forced_alignment_attention2="teacher_forcing_additive",  # teacher_forcing_forward, teacher_forcing_additive
    attention1_out_units=224,
    attention2_out_units=32,

    ## Decoder V1
    decoder_prenet_drop_rate=0.5,
    decoder_prenet_out_units=(256, 128),
    attention_out_units=256,
    decoder_out_units=256,

    ## Decoder V2
    attention_kernel=31,
    attention_filters=32,
    cumulative_weights=False,

    ## Forward attention
    use_forward_attention_transition_agent=False,

    ## Decoder Self Attention
    decoder_self_attention_out_units=256,
    decoder_self_attention_num_heads=2,
    decoder_self_attention_num_hop=1,
    decoder_self_attention_drop_rate=0.05,

    ## Speaker Embedding
    use_speaker_embedding=False,
    num_speakers=1,
    speaker_embedding_dim=16,
    speaker_embedding_offset=0,

    ## Post net
    post_net_cbhg_out_units=256,
    post_net_conv_channels=128,
    post_net_max_filter_width=8,
    post_net_projection1_out_channels=256,
    post_net_projection2_out_channels=80,
    post_net_num_highway=4,

    ## Post net V2
    use_postnet_v2=False,
    num_postnet_v2_layers=5,
    postnet_v2_kernel_size=5,
    postnet_v2_out_channels=512,
    postnet_v2_drop_rate=0.5,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    learning_rate_step_factor=1,
    save_summary_steps=100,
    save_checkpoints_steps=500,
    keep_checkpoint_max=200,
    keep_checkpoint_every_n_hours=1,  # deprecated
    log_step_count_steps=1,
    alignment_save_steps=1000,
    save_training_time_metrics=True,
    approx_min_target_length=100,
    suffle_buffer_size=64,
    batch_bucket_width=50,
    batch_num_buckets=50,
    interleave_cycle_length_cpu_factor=1.0,
    interleave_cycle_length_min=4,
    interleave_cycle_length_max=16,
    interleave_buffer_output_elements=200,
    interleave_prefetch_input_elements=200,
    prefetch_buffer_size=4,
    use_cache=False,
    cache_file_name="",
    logfile="log.txt",
    record_profile=False,
    profile_steps=50,

    # Eval:
    max_iters=500,
    num_evaluation_steps=64,
    eval_start_delay_secs=120,
    eval_throttle_secs=600,

    # Predict
    use_forced_alignment_mode=False,

    # Extention
    use_zoneout_at_encoder=False,
    decoder_version="v1",
    zoneout_factor_cell=0.1,
    zoneout_factor_output=0.1,

    # Pre-process
    trim_top_db=30,
    trim_frame_length=1024,
    trim_hop_length=256,
    num_silent_frames=4,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
