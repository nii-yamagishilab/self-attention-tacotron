# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import AttentionWrapper
from modules.self_attention import SelfAttention
from modules.rnn_wrappers import RNNStateHistoryWrapper, TransformerWrapper, \
    OutputMgcLf0AndStopTokenWrapper, DecoderMgcLf0PreNetWrapper, OutputAndStopTokenTransparentWrapper, \
    OutputMgcLf0AndStopTokenTransparentWrapper
from modules.helpers import TransformerTrainingHelper, TrainingMgcLf0Helper, ValidationMgcLf0Helper, \
    StopTokenBasedMgcLf0InferenceHelper
from modules.multi_speaker_modules import MultiSpeakerPreNet
from tacotron2.tacotron.modules import PreNet, CBHG, Conv1d, HighwayNet, ZoneoutLSTMCell
from tacotron2.tacotron.tacotron_v1 import DecoderRNNV1
from tacotron2.tacotron.tacotron_v2 import DecoderRNNV2
from tacotron2.tacotron.rnn_wrappers import OutputAndStopTokenWrapper, AttentionRNN, ConcatOutputAndAttentionWrapper, \
    DecoderPreNetWrapper
from tacotron2.tacotron.helpers import StopTokenBasedInferenceHelper, TrainingHelper, ValidationHelper
from tacotron2.tacotron.rnn_impl import LSTMImpl
from functools import reduce
from typing import Tuple


class ZoneoutCBHG(tf.layers.Layer):

    def __init__(self, out_units, conv_channels, max_filter_width, projection1_out_channels, projection2_out_channels,
                 num_highway, is_training,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0, lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        super(ZoneoutCBHG, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self.out_units = out_units
        self._is_training = is_training
        self._zoneout_factor_cell = zoneout_factor_cell
        self._zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   is_training=is_training,
                   name=f"conv1d_K{kernel_size}",
                   dtype=dtype)
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = tf.layers.MaxPooling1D(pool_size=2, strides=1, padding="SAME", dtype=dtype)

        self.projection1 = Conv1d(kernel_size=3,
                                  out_channels=projection1_out_channels,
                                  activation=tf.nn.relu,
                                  is_training=is_training,
                                  name="proj1",
                                  dtype=dtype)

        self.projection2 = Conv1d(kernel_size=3,
                                  out_channels=projection2_out_channels,
                                  activation=tf.identity,
                                  is_training=is_training,
                                  name="proj2",
                                  dtype=dtype)

        self.adjustment_layer = tf.layers.Dense(half_out_units, dtype=dtype)

        self.highway_nets = [HighwayNet(half_out_units, dtype=dtype) for i in range(1, num_highway + 1)]

    def build(self, _):
        self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            ZoneoutLSTMCell(self.out_units // 2,
                            self._is_training,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            lstm_impl=self._lstm_impl,
                            dtype=self.dtype),
            ZoneoutLSTMCell(self.out_units // 2,
                            self._is_training,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            lstm_impl=self._lstm_impl,
                            dtype=self.dtype),
            highway_output,
            sequence_length=input_lengths,
            dtype=highway_output.dtype)

        return tf.concat(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.out_units])


class SelfAttentionCBHG(tf.layers.Layer):

    def __init__(self, out_units,
                 conv_channels,
                 max_filter_width,
                 projection1_out_channels,
                 projection2_out_channels,
                 num_highway, self_attention_out_units,
                 self_attention_num_heads,
                 is_training,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0, self_attention_drop_rate=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        half_out_units = out_units // 2
        assert out_units % 2 == 0
        assert num_highway == 4
        super(SelfAttentionCBHG, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self.out_units = out_units
        self._is_training = is_training
        self._zoneout_factor_cell = zoneout_factor_cell
        self._zoneout_factor_output = zoneout_factor_output
        self._self_attention_out_units = self_attention_out_units
        self._lstm_impl = lstm_impl

        self.convolution_banks = [
            Conv1d(kernel_size,
                   conv_channels,
                   activation=tf.nn.relu,
                   is_training=is_training,
                   name=f"conv1d_K{kernel_size}",
                   dtype=dtype)
            for kernel_size in range(1, max_filter_width + 1)]
        self.maxpool = tf.layers.MaxPooling1D(pool_size=2, strides=1, padding="SAME", dtype=dtype)

        self.projection1 = Conv1d(kernel_size=3,
                                  out_channels=projection1_out_channels,
                                  activation=tf.nn.relu,
                                  is_training=is_training,
                                  name="proj1",
                                  dtype=dtype)

        self.projection2 = Conv1d(kernel_size=3,
                                  out_channels=projection2_out_channels,
                                  activation=tf.identity,
                                  is_training=is_training,
                                  name="proj2",
                                  dtype=dtype)

        self.adjustment_layer = tf.layers.Dense(half_out_units, dtype=dtype)

        self.highway_nets = [HighwayNet(half_out_units, dtype=dtype) for i in range(1, num_highway + 1)]

        self.self_attention_adjustment_layer = tf.layers.Dense(self_attention_out_units, dtype=dtype)

        self.self_attention_highway_nets = [HighwayNet(self_attention_out_units, dtype=dtype)
                                            for i in range(1, num_highway + 1)]

        self.self_attention = SelfAttention(self_attention_out_units, self_attention_num_heads,
                                            self_attention_drop_rate, is_training, dtype=dtype)

    def build(self, _):
        self.built = True

    def call(self, inputs, input_lengths=None, positional_encoding=None, **kwargs):
        conv_outputs = tf.concat([conv1d(inputs) for conv1d in self.convolution_banks], axis=-1)

        maxpool_output = self.maxpool(conv_outputs)

        proj1_output = self.projection1(maxpool_output)
        proj2_output = self.projection2(proj1_output)

        # residual connection
        highway_input = proj2_output + inputs

        if highway_input.shape[2] != self.out_units // 2:
            highway_input = self.adjustment_layer(highway_input)

        highway_output = reduce(lambda acc, hw: hw(acc), self.highway_nets, highway_input)

        self_attention_highway_input = self.self_attention_adjustment_layer(highway_input)

        self_attention_highway_output = reduce(lambda acc, hw: hw(acc), self.self_attention_highway_nets,
                                               self_attention_highway_input)

        self_attention_input = self_attention_highway_output + positional_encoding

        self_attention_output, self_attention_alignments = self.self_attention(self_attention_input,
                                                                               memory_sequence_length=input_lengths)
        self_attention_output = self_attention_output + self_attention_highway_output

        bilstm_outputs, bilstm_states = tf.nn.bidirectional_dynamic_rnn(
            ZoneoutLSTMCell(self.out_units // 2,
                            self._is_training,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            dtype=self.dtype),
            ZoneoutLSTMCell(self.out_units // 2,
                            self._is_training,
                            zoneout_factor_cell=self._zoneout_factor_cell,
                            zoneout_factor_output=self._zoneout_factor_output,
                            dtype=self.dtype),
            highway_output,
            sequence_length=input_lengths,
            dtype=highway_output.dtype)

        bilstm_outputs = tf.concat(bilstm_outputs, axis=-1)
        return bilstm_outputs, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return (tf.TensorShape([input_shape[0], input_shape[1], self.out_units]),
                tf.TensorShape([input_shape[0], input_shape[1], self.self._self_attention_out_units]))


class EncoderV1WithAccentType(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=256, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 prenet_out_units=(224, 112),
                 accent_type_prenet_out_units=(32, 16),
                 drop_rate=0.5,
                 use_zoneout=False,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(EncoderV1WithAccentType, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.prenet_out_units = prenet_out_units
        self.accent_type_prenet_out_units = accent_type_prenet_out_units
        self.cbhg_out_units = cbhg_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in prenet_out_units]
        self.accent_type_prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in
                                    accent_type_prenet_out_units]

        self.cbhg = ZoneoutCBHG(cbhg_out_units,
                                conv_channels,
                                max_filter_width,
                                projection1_out_channels,
                                projection2_out_channels,
                                num_highway,
                                is_training,
                                zoneout_factor_cell,
                                zoneout_factor_output,
                                lstm_impl=lstm_impl,
                                dtype=dtype) if use_zoneout else CBHG(cbhg_out_units,
                                                                      conv_channels,
                                                                      max_filter_width,
                                                                      projection1_out_channels,
                                                                      projection2_out_channels,
                                                                      num_highway,
                                                                      is_training,
                                                                      dtype=dtype)

    def build(self, input_shape):
        (phoneme_input_shape, accent_type_shape) = input_shape
        embed_dim = phoneme_input_shape[2].value
        accent_type_embed_dim = accent_type_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim),
                                      tf.assert_equal(self.accent_type_prenet_out_units[0], accent_type_embed_dim),
                                      tf.assert_equal(self.cbhg_out_units, embed_dim + accent_type_embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        input, accent_type = inputs
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, input)
        accent_type_prenet_output = reduce(lambda acc, pn: pn(acc), self.accent_type_prenets, accent_type)
        concatenated = tf.concat([prenet_output, accent_type_prenet_output], axis=-1)
        cbhg_output = self.cbhg(concatenated, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class ZoneoutEncoderV1(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=256, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 prenet_out_units=(256, 128),
                 drop_rate=0.5,
                 use_zoneout=False,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(ZoneoutEncoderV1, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.prenet_out_units = prenet_out_units
        self.cbhg_out_units = cbhg_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in prenet_out_units]

        self.cbhg = ZoneoutCBHG(cbhg_out_units,
                                conv_channels,
                                max_filter_width,
                                projection1_out_channels,
                                projection2_out_channels,
                                num_highway,
                                is_training,
                                zoneout_factor_cell,
                                zoneout_factor_output,
                                lstm_impl=lstm_impl,
                                dtype=dtype) if use_zoneout else CBHG(cbhg_out_units,
                                                                      conv_channels,
                                                                      max_filter_width,
                                                                      projection1_out_channels,
                                                                      projection2_out_channels,
                                                                      num_highway,
                                                                      is_training,
                                                                      dtype=dtype)

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        cbhg_output = self.cbhg(prenet_output, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class SelfAttentionTransformer(tf.layers.Layer):

    def __init__(self, is_training, out_units=32, num_conv_layers=1, kernel_size=5, self_attention_out_units=256,
                 self_attention_num_heads=2, self_attention_drop_rate=0.05,
                 use_subsequent_mask=False,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SelfAttentionTransformer, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self.self_attention = SelfAttention(self_attention_out_units, self_attention_num_heads,
                                            self_attention_drop_rate, is_training,
                                            use_subsequent_mask=use_subsequent_mask,
                                            dtype=dtype)

        self.transform_layers = [tf.layers.Dense(out_units, activation=tf.nn.tanh, dtype=dtype)]

    def build(self, _):
        self.built = True

    def call(self, inputs, memory_sequence_length=None):
        self_attention_output, self_attention_alignment = self.self_attention(inputs,
                                                                              memory_sequence_length=memory_sequence_length)

        transformed = reduce(lambda acc, l: l(acc), self.transform_layers, self_attention_output)

        residual = inputs + transformed

        return residual, self_attention_alignment


class SelfAttentionCBHGEncoder(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=224, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 self_attention_out_units=32,
                 self_attention_num_heads=2,
                 self_attention_num_hop=1,
                 self_attention_transformer_num_conv_layers=1,
                 self_attention_transformer_kernel_size=5,
                 prenet_out_units=(256, 128), drop_rate=0.5,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 self_attention_drop_rate=0.1,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SelfAttentionCBHGEncoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.prenet_out_units = prenet_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in prenet_out_units]

        self.cbhg = ZoneoutCBHG(cbhg_out_units,
                                conv_channels,
                                max_filter_width,
                                projection1_out_channels,
                                projection2_out_channels,
                                num_highway,
                                is_training,
                                zoneout_factor_cell,
                                zoneout_factor_output,
                                lstm_impl=lstm_impl,
                                dtype=dtype)

        self.self_attention_projection_layer = tf.layers.Dense(self_attention_out_units, dtype=dtype)

        self.self_attention = [
            SelfAttentionTransformer(is_training,
                                     out_units=self_attention_out_units,
                                     self_attention_out_units=self_attention_out_units,
                                     self_attention_num_heads=self_attention_num_heads,
                                     self_attention_drop_rate=self_attention_drop_rate,
                                     use_subsequent_mask=False,
                                     dtype=dtype) for i in
            range(1, self_attention_num_hop + 1)]

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        lstm_output = self.cbhg(prenet_output, input_lengths=input_lengths)

        self_attention_input = self.self_attention_projection_layer(lstm_output)

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=input_lengths)
            return output, alignments + alignment

        self_attention_output, self_attention_alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                                                  self.self_attention,
                                                                  (self_attention_input, []))
        return lstm_output, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class SelfAttentionCBHGEncoderWithAccentType(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=224, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 self_attention_out_units=32,
                 self_attention_num_heads=2,
                 self_attention_num_hop=1,
                 self_attention_transformer_num_conv_layers=1,
                 self_attention_transformer_kernel_size=5,
                 prenet_out_units=(224, 112),
                 accent_type_prenet_out_units=(32, 16),
                 drop_rate=0.5,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 self_attention_drop_rate=0.1,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SelfAttentionCBHGEncoderWithAccentType, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                                                     **kwargs)
        self.prenet_out_units = prenet_out_units
        self.accent_type_prenet_out_units = accent_type_prenet_out_units
        self.self_attention_out_units = self_attention_out_units
        self.cbhg_out_units = cbhg_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in prenet_out_units]
        self.accent_type_prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in
                                    accent_type_prenet_out_units]

        self.cbhg = ZoneoutCBHG(cbhg_out_units,
                                conv_channels,
                                max_filter_width,
                                projection1_out_channels,
                                projection2_out_channels,
                                num_highway,
                                is_training,
                                zoneout_factor_cell,
                                zoneout_factor_output,
                                lstm_impl=lstm_impl,
                                dtype=dtype)

        self.self_attention_projection_layer = tf.layers.Dense(self_attention_out_units, dtype=dtype)

        self.self_attention = [
            SelfAttentionTransformer(is_training,
                                     out_units=self_attention_out_units,
                                     self_attention_out_units=self_attention_out_units,
                                     self_attention_num_heads=self_attention_num_heads,
                                     self_attention_drop_rate=self_attention_drop_rate,
                                     use_subsequent_mask=False,
                                     dtype=dtype) for i in
            range(1, self_attention_num_hop + 1)]

    def build(self, input_shape):
        (phoneme_input_shape, accent_type_shape) = input_shape
        embed_dim = phoneme_input_shape[2].value
        accent_type_embed_dim = accent_type_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim),
                                      tf.assert_equal(self.accent_type_prenet_out_units[0], accent_type_embed_dim),
                                      tf.assert_equal(self.cbhg_out_units + self.self_attention_out_units,
                                                      embed_dim + accent_type_embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        input, accent_type = inputs
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, input)
        accent_type_prenet_output = reduce(lambda acc, pn: pn(acc), self.accent_type_prenets, accent_type)
        concatenated = tf.concat([prenet_output, accent_type_prenet_output], axis=-1)
        lstm_output = self.cbhg(concatenated, input_lengths=input_lengths)

        self_attention_input = self.self_attention_projection_layer(lstm_output)

        def self_attend(input, alignments, layer):
            output, alignment = layer(input, memory_sequence_length=input_lengths)
            return output, alignments + alignment

        self_attention_output, self_attention_alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                                                  self.self_attention,
                                                                  (self_attention_input, []))
        return lstm_output, self_attention_output, self_attention_alignments

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


class ExtendedDecoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(ExtendedDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_out_units = attention_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, source, attention_fn=None, speaker_embed=None, is_training=None, is_validation=None,
             teacher_forcing=False, memory_sequence_length=None, target_sequence_length=None,
             target=None, teacher_alignments=None, apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention_fn is not None

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                       PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference,
                              dtype=self.dtype))
        else:
            prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                    dtype=self.dtype)
                             for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source)[0]
        attention_mechanism = attention_fn(source, memory_sequence_length, teacher_alignments)
        attention_cell = AttentionRNN(ZoneoutLSTMCell(self.attention_out_units,
                                                      is_training,
                                                      zoneout_factor_cell=self.zoneout_factor_cell,
                                                      zoneout_factor_output=self.zoneout_factor_output,
                                                      lstm_impl=self._lstm_impl,
                                                      dtype=self.dtype),
                                      prenets,
                                      attention_mechanism,
                                      dtype=self.dtype)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            zoneout_factor_cell=self.zoneout_factor_cell,
            zoneout_factor_output=self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None
        output_and_done_cell = OutputAndStopTokenWrapper(decoder_cell, self.num_mels * self.outputs_per_step,
                                                         dtype=self.dtype)

        decoder_initial_state = output_and_done_cell.zero_state(batch_size, dtype=source.dtype)

        helper = TrainingHelper(target,
                                self.num_mels,
                                self.outputs_per_step,
                                n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationHelper(target, batch_size,
                                  self.num_mels,
                                  self.outputs_per_step,
                                  n_feed_frame=self.n_feed_frame,
                                  teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedInferenceHelper(batch_size,
                                               self.num_mels,
                                               self.outputs_per_step,
                                               n_feed_frame=self.n_feed_frame,
                                               dtype=source.dtype)

        ((decoder_outputs, stop_token), _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_and_done_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class Projection:

    def __init__(self, in_units, out_units, is_training, dtype=None, name="projection"):
        with tf.variable_scope(name):
            self.kernel = tf.get_variable('kernel',
                                          shape=[in_units, out_units],
                                          dtype=dtype)

            self.bias = tf.get_variable('bias',
                                        shape=[out_units, ],
                                        dtype=dtype,
                                        initializer=tf.zeros_initializer(dtype=dtype))

    def __call__(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        matmul = tf.tensordot(inputs, self.kernel, axes=[[len(shape) - 1], [0]])
        output = tf.nn.bias_add(matmul, self.bias)
        return output


class MGCProjection:

    def __init__(self, in_units, out_units, is_training, dtype=None, name="mgc_projection"):
        with tf.variable_scope(name):
            self.dense_kernel1 = tf.get_variable('dense_kernel1',
                                                 shape=[in_units, in_units],
                                                 dtype=dtype)

            self.dense_bias1 = tf.get_variable('dense_bias1',
                                               shape=[in_units, ],
                                               dtype=dtype,
                                               initializer=tf.zeros_initializer(dtype=dtype))

            self.dense_kernel2 = tf.get_variable('dense_kernel2',
                                                 shape=[in_units, out_units],
                                                 dtype=dtype)

            self.dense_bias2 = tf.get_variable('dense_bias2',
                                               shape=[out_units, ],
                                               dtype=dtype,
                                               initializer=tf.zeros_initializer(dtype=dtype))

    def __call__(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()
        matmul1 = tf.tensordot(inputs, self.dense_kernel1, axes=[[len(shape) - 1], [0]])
        dense_output1 = tf.nn.bias_add(matmul1, self.dense_bias1)
        dense_output1 = tf.nn.tanh(dense_output1)
        matmul2 = tf.tensordot(dense_output1, self.dense_kernel2, axes=[[len(shape) - 1], [0]])
        dense_output2 = tf.nn.bias_add(matmul2, self.dense_bias2)
        return dense_output2


class RNNTransformer:

    def __init__(self, is_training, decoder_cell, decoder_initial_state,
                 self_attention_out_units,
                 self_attention_transformer_num_conv_layers,
                 self_attention_transformer_kernel_size,
                 self_attention_num_heads,
                 self_attention_num_hop,
                 self_attention_drop_rate,
                 num_mels,
                 outputs_per_step,
                 n_feed_frame,
                 max_iters,
                 batch_size,
                 dtype,
                 output_dtype):
        self._is_training = is_training
        self.decoder_cell = decoder_cell
        self.decoder_initial_state = decoder_initial_state
        self._out_units = num_mels * outputs_per_step
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.n_feed_frame = n_feed_frame
        self.max_iters = max_iters
        self._batch_size = batch_size
        self._dtype = dtype
        self._output_dtype = output_dtype

        self.transformers = [
            SelfAttentionTransformer(is_training,
                                     out_units=self_attention_out_units,
                                     self_attention_out_units=self_attention_out_units,
                                     self_attention_num_heads=self_attention_num_heads,
                                     self_attention_drop_rate=self_attention_drop_rate,
                                     use_subsequent_mask=True,
                                     dtype=dtype) for i in
            range(1, self_attention_num_hop + 1)]

        # at inference time, outputs are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
        with tf.variable_scope("decoder") as decoder_scope:
            self.out_projection = Projection(self_attention_out_units, self._out_units, is_training,
                                             dtype=self._dtype,
                                             name="out_projection")
            self.stop_token_projection = Projection(self_attention_out_units, 1, is_training,
                                                    dtype=self._dtype,
                                                    name="stop_token_projection")
            self._decoder_scope = decoder_scope

    def __call__(self, target=None, is_training=None, is_validation=None, teacher_forcing=False,
                 memory_sequence_length=None):
        helper = TransformerTrainingHelper(target,
                                           self.num_mels,
                                           self.outputs_per_step,
                                           n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationHelper(target, self._batch_size,
                                  self.num_mels,
                                  self.outputs_per_step,
                                  n_feed_frame=self.n_feed_frame,
                                  teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedInferenceHelper(self._batch_size,
                                               self.num_mels,
                                               self.outputs_per_step,
                                               n_feed_frame=self.n_feed_frame,
                                               dtype=self._output_dtype)

        if is_training:
            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state),
                maximum_iterations=self.max_iters,
                scope=self._decoder_scope)

            def self_attend(input, alignments, layer):
                output, alignment = layer(input, memory_sequence_length=memory_sequence_length)
                return output, alignments + alignment

            # at inference time, transformers are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:
                transformed, alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                                 self.transformers,
                                                 (decoder_outputs, []))
            mel_output = self.out_projection(transformed)
            stop_token = self.stop_token_projection(transformed)
            return mel_output, stop_token, final_decoder_state

        else:
            transformer_cell = TransformerWrapper(
                RNNStateHistoryWrapper(self.decoder_cell, self.max_iters),
                self.transformers, memory_sequence_length)
            output_and_done_cell = OutputAndStopTokenTransparentWrapper(transformer_cell, self._out_units,
                                                                        self.out_projection,
                                                                        self.stop_token_projection)

            decoder_initial_state = output_and_done_cell.zero_state(self._batch_size, dtype=self._output_dtype)

            ((decoder_outputs, stop_token), _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(output_and_done_cell, helper, decoder_initial_state),
                scope=self._decoder_scope,
                maximum_iterations=self.max_iters,
                parallel_iterations=10,
                swap_memory=True)  # Huge memory consumption at inference time
            return decoder_outputs, stop_token, final_decoder_state


class MgcLf0RNNTransformer:

    def __init__(self, is_training, decoder_cell, decoder_initial_state,
                 self_attention_out_units,
                 self_attention_transformer_num_conv_layers,
                 self_attention_transformer_kernel_size,
                 self_attention_num_heads,
                 self_attention_num_hop,
                 self_attention_drop_rate,
                 num_mgcs,
                 num_lf0s,
                 outputs_per_step,
                 n_feed_frame,
                 max_iters,
                 batch_size,
                 dtype,
                 output_dtype):
        self._is_training = is_training
        self.decoder_cell = decoder_cell
        self.decoder_initial_state = decoder_initial_state
        self._mgc_out_units = num_mgcs * outputs_per_step
        self._lf0_out_units = num_lf0s * outputs_per_step
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        self.outputs_per_step = outputs_per_step
        self.n_feed_frame = n_feed_frame
        self.max_iters = max_iters
        self._batch_size = batch_size
        self._dtype = dtype
        self._output_dtype = output_dtype

        self.transformers = [
            SelfAttentionTransformer(is_training,
                                     out_units=self_attention_out_units,
                                     self_attention_out_units=self_attention_out_units,
                                     self_attention_num_heads=self_attention_num_heads,
                                     self_attention_drop_rate=self_attention_drop_rate,
                                     use_subsequent_mask=True,
                                     dtype=dtype) for i in
            range(1, self_attention_num_hop + 1)]

        # at inference time, outputs are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
        with tf.variable_scope("decoder") as decoder_scope:
            self.mgc_out_projection = MGCProjection(self_attention_out_units, self._mgc_out_units, is_training,
                                                    dtype=self._dtype,
                                                    name="mgc_out_projection")
            self.lf0_out_projection = Projection(self_attention_out_units, self._lf0_out_units, is_training,
                                                 dtype=self._dtype,
                                                 name="lf0_out_projection")
            self.stop_token_projection = Projection(self_attention_out_units, 1, is_training,
                                                    dtype=self._dtype,
                                                    name="stop_token_projection")
            self._decoder_scope = decoder_scope

    def __call__(self, target=None, is_training=None, is_validation=None, teacher_forcing=False,
                 memory_sequence_length=None):
        mgc_targets, lf0_targets = target
        helper = TrainingMgcLf0Helper(mgc_targets,
                                      lf0_targets,
                                      self.num_mgcs,
                                      self.num_lf0s,
                                      self.outputs_per_step,
                                      n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationMgcLf0Helper(mgc_targets,
                                        lf0_targets, self._batch_size,
                                        self.num_mgcs,
                                        self.num_lf0s,
                                        self.outputs_per_step,
                                        n_feed_frame=self.n_feed_frame,
                                        teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedMgcLf0InferenceHelper(self._batch_size,
                                                     self.num_mgcs,
                                                     self.num_lf0s,
                                                     self.outputs_per_step,
                                                     n_feed_frame=self.n_feed_frame,
                                                     dtype=self._output_dtype)

        if is_training:
            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state),
                maximum_iterations=self.max_iters,
                scope=self._decoder_scope)

            def self_attend(input, alignments, layer):
                output, alignment = layer(input, memory_sequence_length=memory_sequence_length)
                return output, alignments + alignment

            # at inference time, transformers are evaluated within dynamic_decode (dynamic_decode has "decoder" scope)
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:
                transformed, alignments = reduce(lambda acc, sa: self_attend(acc[0], acc[1], sa),
                                                 self.transformers,
                                                 (decoder_outputs, []))
            mgc_output = self.mgc_out_projection(transformed)
            lf0_output = self.lf0_out_projection(transformed)
            stop_token = self.stop_token_projection(transformed)
            return mgc_output, lf0_output, stop_token, final_decoder_state

        else:
            transformer_cell = TransformerWrapper(
                RNNStateHistoryWrapper(self.decoder_cell, self.max_iters),
                self.transformers, memory_sequence_length)
            output_and_done_cell = OutputMgcLf0AndStopTokenTransparentWrapper(transformer_cell,
                                                                              self._mgc_out_units,
                                                                              self._lf0_out_units,
                                                                              self.mgc_out_projection,
                                                                              self.lf0_out_projection,
                                                                              self.stop_token_projection)

            decoder_initial_state = output_and_done_cell.zero_state(self._batch_size, dtype=self._output_dtype)

            ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
             _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(output_and_done_cell, helper, decoder_initial_state),
                scope=self._decoder_scope,
                maximum_iterations=self.max_iters,
                parallel_iterations=10,
                swap_memory=True)  # Huge memory consumption at inference time

            return decoder_mgc_outputs, decoder_lf0_outputs, stop_token, final_decoder_state


class TransformerDecoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 self_attention_out_units=256,
                 self_attention_num_heads=2,
                 self_attention_num_hop=1,
                 self_attention_transformer_num_conv_layers=1,
                 self_attention_transformer_kernel_size=5,
                 self_attention_drop_rate=0.05,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(TransformerDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_out_units = attention_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.self_attention_out_units = self_attention_out_units
        self.self_attention_num_heads = self_attention_num_heads
        self.self_attention_num_hop = self_attention_num_hop
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self.self_attention_drop_rate = self_attention_drop_rate
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, source, attention_fn=None, speaker_embed=None, is_training=None, is_validation=None,
             teacher_forcing=False, memory_sequence_length=None, target_sequence_length=None,
             target=None, teacher_alignments=None,
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention_fn is not None

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                       PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference,
                              dtype=self.dtype))
        else:
            prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                    dtype=self.dtype)
                             for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source)[0]
        attention_mechanism = attention_fn(source, memory_sequence_length, teacher_alignments)
        attention_cell = AttentionRNN(ZoneoutLSTMCell(self.attention_out_units,
                                                      is_training,
                                                      self.zoneout_factor_cell,
                                                      self.zoneout_factor_output,
                                                      lstm_impl=self._lstm_impl,
                                                      dtype=self.dtype),
                                      prenets,
                                      attention_mechanism,
                                      dtype=self.dtype)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=source.dtype)

        rnn_transformer = RNNTransformer(is_training, decoder_cell, decoder_initial_state,
                                         self.self_attention_out_units,
                                         self.self_attention_transformer_num_conv_layers,
                                         self.self_attention_transformer_kernel_size,
                                         self.self_attention_num_heads,
                                         self.self_attention_num_hop,
                                         self.self_attention_drop_rate,
                                         self.num_mels,
                                         self.outputs_per_step,
                                         self.n_feed_frame,
                                         self.max_iters,
                                         batch_size,
                                         self.dtype,
                                         source.dtype)

        decoder_outputs, stop_token, final_decoder_state = rnn_transformer(target, is_training=is_training,
                                                                           is_validation=is_validation,
                                                                           teacher_forcing=teacher_forcing,
                                                                           memory_sequence_length=target_sequence_length)

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class DualSourceAttentionRNN(RNNCell):

    def __init__(self, cell, prenets: Tuple[PreNet],
                 attention_mechanism1,
                 attention_mechanism2,
                 trainable=True, name=None, **kwargs):
        super(DualSourceAttentionRNN, self).__init__(name=name, trainable=trainable, **kwargs)
        attention_cell = AttentionWrapper(
            cell,
            [attention_mechanism1, attention_mechanism2],
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderPreNetWrapper(attention_cell, prenets)
        concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)
        self._cell = concat_cell

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
        return self._cell(inputs, state)


class DualSourceDecoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_rnn_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DualSourceDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_rnn_out_units = attention_rnn_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, sources, attention1_fn=None, attention2_fn=None, speaker_embed=None,
             is_training=None, is_validation=None,
             teacher_forcing=False,
             memory_sequence_length=None, memory2_sequence_length=None,
             target=None, teacher_alignments=(None, None),
             target_sequence_length=None,  # target_sequence_length is not used except TransformerDecoder
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention1_fn is not None
        assert attention2_fn is not None

        source1, source2 = sources

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                       PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
        else:
            prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                    dtype=self.dtype)
                             for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source1)[0]
        attention_mechanism1 = attention1_fn(source1, memory_sequence_length, teacher_alignments[0])
        attention_mechanism2 = attention2_fn(source2, memory2_sequence_length, teacher_alignments[1])
        attention_cell = DualSourceAttentionRNN(ZoneoutLSTMCell(self.attention_rnn_out_units,
                                                                is_training,
                                                                self.zoneout_factor_cell,
                                                                self.zoneout_factor_output,
                                                                lstm_impl=self._lstm_impl,
                                                                dtype=self.dtype),
                                                prenets,
                                                attention_mechanism1,
                                                attention_mechanism2,
                                                dtype=self.dtype)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None
        output_and_done_cell = OutputAndStopTokenWrapper(decoder_cell, self.num_mels * self.outputs_per_step,
                                                         dtype=self.dtype)

        decoder_initial_state = output_and_done_cell.zero_state(batch_size, dtype=source1.dtype)

        helper = TrainingHelper(target,
                                self.num_mels,
                                self.outputs_per_step,
                                n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationHelper(target, batch_size,
                                  self.num_mels,
                                  self.outputs_per_step,
                                  n_feed_frame=self.n_feed_frame,
                                  teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedInferenceHelper(batch_size,
                                               self.num_mels,
                                               self.outputs_per_step,
                                               n_feed_frame=self.n_feed_frame,
                                               dtype=source1.dtype)

        ((decoder_outputs, stop_token), _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_and_done_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class MgcLf0AttentionRNN(RNNCell):

    def __init__(self, cell, mgc_prenets: Tuple[PreNet], lf0_prenets: Tuple[PreNet],
                 attention_mechanism,
                 trainable=True, name=None, **kwargs):
        super(MgcLf0AttentionRNN, self).__init__(name=name, trainable=trainable, **kwargs)
        attention_cell = AttentionWrapper(
            cell,
            attention_mechanism,
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderMgcLf0PreNetWrapper(attention_cell, mgc_prenets, lf0_prenets)
        concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self._cell = concat_cell

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
        return self._cell(inputs, state)


class DualSourceMgcLf0AttentionRNN(RNNCell):

    def __init__(self, cell, mgc_prenets: Tuple[PreNet], lf0_prenets: Tuple[PreNet],
                 attention_mechanism1,
                 attention_mechanism2,
                 trainable=True, name=None, **kwargs):
        super(DualSourceMgcLf0AttentionRNN, self).__init__(name=name, trainable=trainable, **kwargs)
        attention_cell = AttentionWrapper(
            cell,
            [attention_mechanism1, attention_mechanism2],
            alignment_history=True,
            output_attention=False)
        prenet_cell = DecoderMgcLf0PreNetWrapper(attention_cell, mgc_prenets, lf0_prenets)
        concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)
        self._cell = concat_cell

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
        return self._cell(inputs, state)


class MgcLf0Decoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_rnn_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mgcs=60,
                 num_lf0s=256,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MgcLf0Decoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_rnn_out_units = attention_rnn_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1, dtype=dtype)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, source, attention_fn=None, speaker_embed=None, is_training=None, is_validation=None,
             teacher_forcing=False,
             memory_sequence_length=None,
             target=None, target_sequence_length=None, teacher_alignments=None,
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention_fn is not None

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            mgc_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
            lf0_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
        else:
            mgc_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
            lf0_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source)[0]
        attention_mechanism = attention_fn(source, memory_sequence_length, teacher_alignments)
        attention_cell = MgcLf0AttentionRNN(ZoneoutLSTMCell(self.attention_rnn_out_units,
                                                            is_training,
                                                            self.zoneout_factor_cell,
                                                            self.zoneout_factor_output,
                                                            lstm_impl=self._lstm_impl,
                                                            dtype=self.dtype),
                                            mgc_prenets,
                                            lf0_prenets,
                                            attention_mechanism)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None
        output_and_done_cell = OutputMgcLf0AndStopTokenWrapper(decoder_cell,
                                                               self.num_mgcs * self.outputs_per_step,
                                                               self.num_lf0s * self.outputs_per_step,
                                                               dtype=self.dtype)

        decoder_initial_state = output_and_done_cell.zero_state(batch_size, dtype=source.dtype)

        helper = TrainingMgcLf0Helper(target[0],
                                      target[1],
                                      self.num_mgcs,
                                      self.num_lf0s,
                                      self.outputs_per_step,
                                      n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationMgcLf0Helper(target[0],
                                        target[1], batch_size,
                                        self.num_mgcs,
                                        self.num_lf0s,
                                        self.outputs_per_step,
                                        n_feed_frame=self.n_feed_frame,
                                        teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedMgcLf0InferenceHelper(batch_size,
                                                     self.num_mgcs,
                                                     self.num_lf0s,
                                                     self.outputs_per_step,
                                                     n_feed_frame=self.n_feed_frame,
                                                     dtype=source.dtype)

        ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
         _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_and_done_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state


class MgcLf0DualSourceDecoder(tf.layers.Layer):

    def __init__(self,
                 prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_rnn_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mgcs=60,
                 num_lf0s=256,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MgcLf0DualSourceDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_rnn_out_units = attention_rnn_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, sources, attention1_fn=None, attention2_fn=None, speaker_embed=None, is_training=None,
             is_validation=None,
             teacher_forcing=False,
             memory_sequence_length=None, memory2_sequence_length=None,
             target=None, teacher_alignments=(None, None),
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention1_fn is not None
        assert attention2_fn is not None

        source1, source2 = sources

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            mgc_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
            lf0_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
        else:
            mgc_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
            lf0_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source1)[0]
        attention_mechanism1 = attention1_fn(source1, memory_sequence_length, teacher_alignments[0])
        attention_mechanism2 = attention2_fn(source2, memory2_sequence_length, teacher_alignments[1])
        attention_cell = DualSourceMgcLf0AttentionRNN(ZoneoutLSTMCell(self.attention_rnn_out_units,
                                                                      is_training,
                                                                      self.zoneout_factor_cell,
                                                                      self.zoneout_factor_output,
                                                                      lstm_impl=self._lstm_impl,
                                                                      dtype=self.dtype),
                                                      mgc_prenets,
                                                      lf0_prenets,
                                                      attention_mechanism1,
                                                      attention_mechanism2)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None
        output_and_done_cell = OutputMgcLf0AndStopTokenWrapper(decoder_cell,
                                                               self.num_mgcs * self.outputs_per_step,
                                                               self.num_lf0s * self.outputs_per_step,
                                                               dtype=self.dtype)

        decoder_initial_state = output_and_done_cell.zero_state(batch_size, dtype=source1.dtype)

        helper = TrainingMgcLf0Helper(target[0],
                                      target[1],
                                      self.num_mgcs,
                                      self.num_lf0s,
                                      self.outputs_per_step,
                                      n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationMgcLf0Helper(target[0],
                                        target[1], batch_size,
                                        self.num_mgcs,
                                        self.num_lf0s,
                                        self.outputs_per_step,
                                        n_feed_frame=self.n_feed_frame,
                                        teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedMgcLf0InferenceHelper(batch_size,
                                                     self.num_mgcs,
                                                     self.num_lf0s,
                                                     self.outputs_per_step,
                                                     n_feed_frame=self.n_feed_frame,
                                                     dtype=source1.dtype)

        ((decoder_mgc_outputs, decoder_lf0_outputs, stop_token),
         _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_and_done_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state


class DualSourceTransformerDecoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_rnn_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 self_attention_out_units=256,
                 self_attention_num_heads=2,
                 self_attention_num_hop=1,
                 self_attention_transformer_num_conv_layers=1,
                 self_attention_transformer_kernel_size=5,
                 self_attention_drop_rate=0.05,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DualSourceTransformerDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_rnn_out_units = attention_rnn_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1, dtype=dtype)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.self_attention_out_units = self_attention_out_units
        self.self_attention_num_heads = self_attention_num_heads
        self.self_attention_num_hop = self_attention_num_hop
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self.self_attention_drop_rate = self_attention_drop_rate
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, sources, attention1_fn=None, attention2_fn=None, speaker_embed=None, is_training=None,
             is_validation=None,
             teacher_forcing=False, memory_sequence_length=None, memory2_sequence_length=None,
             target_sequence_length=None,
             target=None, teacher_alignments=(None, None),
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention1_fn is not None
        assert attention2_fn is not None

        source1, source2 = sources

        if speaker_embed is not None:
            prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                       PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
        else:
            prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                    dtype=self.dtype)
                             for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source1)[0]
        attention_mechanism1 = attention1_fn(source1, memory_sequence_length, teacher_alignments[0])
        attention_mechanism2 = attention2_fn(source2, memory2_sequence_length, teacher_alignments[1])
        attention_cell = DualSourceAttentionRNN(ZoneoutLSTMCell(self.attention_rnn_out_units,
                                                                is_training,
                                                                self.zoneout_factor_cell,
                                                                self.zoneout_factor_output,
                                                                lstm_impl=self._lstm_impl,
                                                                dtype=self.dtype),
                                                prenets,
                                                attention_mechanism1,
                                                attention_mechanism2)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=source1.dtype)

        rnn_transformer = RNNTransformer(is_training, decoder_cell, decoder_initial_state,
                                         self.self_attention_out_units,
                                         self.self_attention_transformer_num_conv_layers,
                                         self.self_attention_transformer_kernel_size,
                                         self.self_attention_num_heads,
                                         self.self_attention_num_hop,
                                         self.self_attention_drop_rate,
                                         self.num_mels,
                                         self.outputs_per_step,
                                         self.n_feed_frame,
                                         self.max_iters,
                                         batch_size,
                                         self.dtype,
                                         source1.dtype)

        decoder_outputs, stop_token, final_decoder_state = rnn_transformer(target, is_training=is_training,
                                                                           is_validation=is_validation,
                                                                           teacher_forcing=teacher_forcing,
                                                                           memory_sequence_length=target_sequence_length)

        mel_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return mel_output, stop_token, final_decoder_state


class DualSourceMgcLf0TransformerDecoder(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_rnn_out_units=256,
                 decoder_version="v1",  # v1 | v2
                 decoder_out_units=256,
                 num_mgcs=80,
                 num_lf0s=256,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 self_attention_out_units=256,
                 self_attention_num_heads=2,
                 self_attention_num_hop=1,
                 self_attention_transformer_num_conv_layers=1,
                 self_attention_transformer_kernel_size=5,
                 self_attention_drop_rate=0.05,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DualSourceMgcLf0TransformerDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_rnn_out_units = attention_rnn_out_units
        self.decoder_version = decoder_version
        self.decoder_out_units = decoder_out_units
        self.num_mgcs = num_mgcs
        self.num_lf0s = num_lf0s
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1)
        self.n_feed_frame = n_feed_frame
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.self_attention_out_units = self_attention_out_units
        self.self_attention_num_heads = self_attention_num_heads
        self.self_attention_num_hop = self_attention_num_hop
        self.self_attention_transformer_num_conv_layers = self_attention_transformer_num_conv_layers
        self.self_attention_transformer_kernel_size = self_attention_transformer_kernel_size
        self.self_attention_drop_rate = self_attention_drop_rate
        self._lstm_impl = lstm_impl

    def build(self, _):
        self.built = True

    def call(self, sources, attention1_fn=None, attention2_fn=None, speaker_embed=None, is_training=None,
             is_validation=None,
             teacher_forcing=False,
             memory_sequence_length=None, memory2_sequence_length=None,
             target_sequence_length=None, target=None, teacher_alignments=(None, None),
             apply_dropout_on_inference=None):
        assert is_training is not None
        assert attention1_fn is not None
        assert attention2_fn is not None

        source1, source2 = sources

        if speaker_embed is not None:
            # ToDo: support dtype arg for MultiSpeakerPreNet
            mgc_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
            lf0_prenets = (MultiSpeakerPreNet(self._prenet_out_units[0], speaker_embed, is_training, self._drop_rate),
                           PreNet(self._prenet_out_units[1], is_training, self._drop_rate, apply_dropout_on_inference))
        else:
            mgc_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])
            lf0_prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference,
                                        dtype=self.dtype)
                                 for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source1)[0]
        attention_mechanism1 = attention1_fn(source1, memory_sequence_length, teacher_alignments[0])
        attention_mechanism2 = attention2_fn(source2, memory2_sequence_length, teacher_alignments[1])
        attention_cell = DualSourceMgcLf0AttentionRNN(ZoneoutLSTMCell(self.attention_rnn_out_units,
                                                                      is_training,
                                                                      self.zoneout_factor_cell,
                                                                      self.zoneout_factor_output,
                                                                      lstm_impl=self._lstm_impl,
                                                                      dtype=self.dtype),
                                                      mgc_prenets,
                                                      lf0_prenets,
                                                      attention_mechanism1,
                                                      attention_mechanism2)
        decoder_cell = DecoderRNNV1(self.decoder_out_units,
                                    attention_cell,
                                    dtype=self.dtype) if self.decoder_version == "v1" else DecoderRNNV2(
            self.decoder_out_units,
            attention_cell,
            is_training,
            self.zoneout_factor_cell,
            self.zoneout_factor_output,
            lstm_impl=self._lstm_impl,
            dtype=self.dtype) if self.decoder_version == "v2" else None

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=source1.dtype)

        rnn_transformer = MgcLf0RNNTransformer(is_training, decoder_cell, decoder_initial_state,
                                               self.self_attention_out_units,
                                               self.self_attention_transformer_num_conv_layers,
                                               self.self_attention_transformer_kernel_size,
                                               self.self_attention_num_heads,
                                               self.self_attention_num_hop,
                                               self.self_attention_drop_rate,
                                               self.num_mgcs,
                                               self.num_lf0s,
                                               self.outputs_per_step,
                                               self.n_feed_frame,
                                               self.max_iters,
                                               self.dtype,
                                               source1.dtype)

        decoder_mgc_outputs, decoder_lf0_outputs, stop_token, final_decoder_state = rnn_transformer(target,
                                                                                                    is_training=is_training,
                                                                                                    is_validation=is_validation,
                                                                                                    teacher_forcing=teacher_forcing,
                                                                                                    memory_sequence_length=target_sequence_length)

        mgc_output = tf.reshape(decoder_mgc_outputs, [batch_size, -1, self.num_mgcs])
        lf0_output = tf.reshape(decoder_lf0_outputs, [batch_size, -1, self.num_lf0s])
        return mgc_output, lf0_output, stop_token, final_decoder_state
