# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """


import tensorflow as tf
import numpy as np
import uuid
from hypothesis import given, assume, settings, unlimited
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from modules.module import RNNTransformer, MgcLf0RNNTransformer
from modules.rnn_wrappers import DecoderMgcLf0PreNetWrapper


@composite
def target_tensor(draw, batch_size, target_dim=integers(2, 20).filter(lambda x: x % 2 == 0), t_factor=integers(2, 8),
                  r=integers(1, 2), elements=integers(-1, 1)):
    r = draw(r)
    t = draw(t_factor) * r
    target_dim = draw(target_dim)
    c = target_dim
    btc = draw(arrays(dtype=np.float32, shape=[batch_size, t, c], elements=elements))
    target_lengths = np.repeat(t, batch_size)
    return btc, target_lengths, r, target_dim


@composite
def all_args(draw, batch_size=integers(1, 3)):
    bs = draw(batch_size)
    target, target_lengths, _r, _target_dim = draw(target_tensor(bs))
    return target, target_lengths, bs, _target_dim, _r


class TransformerTest(tf.test.TestCase):

    @given(args=all_args())
    def test_equality_between_training_and_inference(self, args):
        tf.set_random_seed(12345)
        target, target_lengths, batch_size, target_dim, r = args
        target = tf.convert_to_tensor(target)

        decoder_cell = tf.nn.rnn_cell.LSTMCell(target_dim * r, state_is_tuple=True)

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)

        with tf.variable_scope(str(uuid.uuid4())):
            transformer = RNNTransformer(False, decoder_cell, decoder_initial_state,
                                         self_attention_out_units=target_dim * r,
                                         self_attention_transformer_num_conv_layers=1,
                                         self_attention_transformer_kernel_size=3,
                                         self_attention_num_heads=2,
                                         self_attention_num_hop=1,
                                         self_attention_drop_rate=0.0,
                                         num_mels=target_dim,
                                         outputs_per_step=r,
                                         n_feed_frame=r,
                                         max_iters=100,
                                         batch_size=batch_size,
                                         dtype=target.dtype,
                                         output_dtype=target.dtype)

            training_output, training_stop_token, training_state = transformer(target, is_training=True,
                                                                               is_validation=False,
                                                                               teacher_forcing=False,
                                                                               memory_sequence_length=target_lengths)

            inference_output, inference_stop_token, inference_state = transformer(target, is_training=False,
                                                                                  is_validation=True,
                                                                                  teacher_forcing=True,
                                                                                  memory_sequence_length=target_lengths)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            training_output_value, inference_output_value = sess.run([training_output, inference_output])
            training_stop_token_value, inference_stop_token_value = sess.run(
                [training_stop_token, inference_stop_token])
            self.assertAllClose(training_output_value, inference_output_value)
            self.assertAllClose(training_stop_token_value, inference_stop_token_value)

    @given(args=all_args())
    def test_equality_between_training_and_inference_of_mgc_f0(self, args):
        tf.set_random_seed(12345)
        target, target_lengths, batch_size, target_dim, r = args
        target = tf.convert_to_tensor(target)

        decoder_cell = tf.nn.rnn_cell.LSTMCell(target_dim * r, state_is_tuple=True)

        decoder_cell = DecoderMgcLf0PreNetWrapper(decoder_cell, tuple([tf.identity]), tuple([tf.identity]))

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)

        with tf.variable_scope(str(uuid.uuid4())):
            transformer = MgcLf0RNNTransformer(False, decoder_cell, decoder_initial_state,
                                               self_attention_out_units=target_dim * r,
                                               self_attention_transformer_num_conv_layers=1,
                                               self_attention_transformer_kernel_size=3,
                                               self_attention_num_heads=2,
                                               self_attention_num_hop=1,
                                               self_attention_drop_rate=0.0,
                                               num_mgcs=target_dim,
                                               num_lf0s=target_dim,
                                               outputs_per_step=r,
                                               n_feed_frame=r,
                                               max_iters=100,
                                               batch_size=batch_size,
                                               dtype=target.dtype,
                                               output_dtype=target.dtype)

            training_output1, training_output2, training_stop_token, training_state = transformer((target, target),
                                                                                                  is_training=True,
                                                                                                  is_validation=False,
                                                                                                  teacher_forcing=False,
                                                                                                  memory_sequence_length=target_lengths)

            inference_output1, inference_output2, inference_stop_token, inference_state = transformer((target, target),
                                                                                                      is_training=False,
                                                                                                      is_validation=True,
                                                                                                      teacher_forcing=True,
                                                                                                      memory_sequence_length=target_lengths)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            training_output_value1, training_output_value2, inference_output_value1, inference_output_value2 = sess.run(
                [training_output1,
                 training_output2,
                 inference_output1,
                 inference_output2])
            training_stop_token_value, inference_stop_token_value = sess.run(
                [training_stop_token, inference_stop_token])
            self.assertAllClose(training_output_value1, inference_output_value1)
            self.assertAllClose(training_output_value2, inference_output_value2)
            self.assertAllClose(training_stop_token_value, inference_stop_token_value)


if __name__ == '__main__':
    tf.test.main()
