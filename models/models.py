import tensorflow as tf
from tacotron2.tacotron.modules import Embedding
from tacotron2.tacotron.tacotron_v2 import PostNetV2, EncoderV2
from tacotron2.tacotron.hooks import MetricsSaver
from tacotron2.tacotron.losses import spec_loss, classification_loss, binary_loss
from modules.module import ZoneoutEncoderV1, ExtendedDecoder, EncoderV1WithAccentType, \
    SelfAttentionCBHGEncoder, DualSourceDecoder, TransformerDecoder, \
    DualSourceTransformerDecoder, SelfAttentionCBHGEncoderWithAccentType, \
    MgcLf0Decoder, MgcLf0DualSourceDecoder, DualSourceMgcLf0TransformerDecoder
from modules.metrics import MgcLf0MetricsSaver
from modules.regularizers import l2_regularization_loss
from models.attention_factories import attention_factory, dual_source_attention_factory, \
    force_alignment_attention_factory, \
    force_alignment_dual_source_attention_factory
from multi_speaker_tacotron.modules.external_embedding import ExternalEmbedding
from multi_speaker_tacotron.modules.multi_speaker_postnet import MultiSpeakerPostNet
from multi_speaker_tacotron.modules.channel_encoder_postnet import ChannelEncoderPostNet


class ExtendedTacotronV1Model(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset)

            encoder = encoder_factory(params, is_training)

            decoder = decoder_factory(params)

            if params.use_speaker_embedding:
                speaker_embedding = Embedding(params.num_speakers,
                                              embedding_dim=params.speaker_embedding_dim,
                                              index_offset=params.speaker_embedding_offset)

            target = labels.mel if (is_training or is_validation) else features.mel

            embedding_output = embedding(features.source)
            encoder_output = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            speaker_embedding_output = speaker_embedding(features.speaker_id) if params.use_speaker_embedding else None

            mel_output, stop_token, decoder_state = decoder(encoder_output,
                                                            attention_fn=attention_factory(params),
                                                            speaker_embed=speaker_embedding_output,
                                                            is_training=is_training,
                                                            is_validation=is_validation or params.use_forced_alignment_mode,
                                                            teacher_forcing=params.use_forced_alignment_mode,
                                                            memory_sequence_length=features.source_length,
                                                            target_sequence_length=labels.target_length if is_training else None,
                                                            target=target,
                                                            apply_dropout_on_inference=params.apply_dropout_on_inference)

            if params.decoder == "TransformerDecoder" and not is_training:
                decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
                alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                decoder_rnn_state = decoder_state[0]
                alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_forced_alignment_mode:
                mel_output, stop_token, decoder_state = decoder(encoder_output,
                                                                attention_fn=force_alignment_attention_factory(params),
                                                                speaker_embed=speaker_embedding_output,
                                                                is_training=is_training,
                                                                is_validation=True,
                                                                teacher_forcing=False,
                                                                memory_sequence_length=features.source_length,
                                                                target_sequence_length=labels.target_length if is_training else None,
                                                                target=target,
                                                                teacher_alignments=tf.transpose(alignment, [0, 2, 1]),
                                                                apply_dropout_on_inference=params.apply_dropout_on_inference)
                if params.decoder == "TransformerDecoder" and not is_training:
                    alignment = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history.stack(), [1, 2, 0])
                else:
                    alignment = tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])

            if params.use_postnet_v2:
                postnet = PostNetV2(out_units=params.num_mels,
                                    num_postnet_layers=params.num_postnet_v2_layers,
                                    kernel_size=params.postnet_v2_kernel_size,
                                    out_channels=params.postnet_v2_out_channels,
                                    is_training=is_training,
                                    drop_rate=params.postnet_v2_drop_rate)

                postnet_v2_mel_output = postnet(mel_output)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                mel_loss = spec_loss(mel_output, labels.mel,
                                     labels.spec_loss_mask, params.spec_loss_type)
                done_loss = binary_loss(stop_token, labels.done, labels.binary_loss_mask)

                blacklist = ["embedding", "bias", "batch_normalization", "output_projection_wrapper/kernel",
                             "lstm_cell",
                             "output_and_stop_token_wrapper/dense/", "output_and_stop_token_wrapper/dense_1/"]
                regularization_loss = l2_regularization_loss(
                    tf.trainable_variables(), params.l2_regularization_weight,
                    blacklist) if params.use_l2_regularization else 0

                postnet_v2_mel_loss = spec_loss(postnet_v2_mel_output, labels.mel,
                                                labels.spec_loss_mask, params.spec_loss_type) if params.use_postnet_v2 else 0
                loss = mel_loss + done_loss + regularization_loss + postnet_v2_mel_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step,
                    params.learning_rate_step_factor) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, mel_loss, done_loss, lr, postnet_v2_mel_loss, regularization_loss)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MetricsSaver([alignment], global_step, mel_output, labels.mel,
                                                   labels.target_length,
                                                   features.id,
                                                   features.text,
                                                   params.alignment_save_steps,
                                                   mode, summary_writer,
                                                   save_training_time_metrics=params.save_training_time_metrics,
                                                   keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                mel_output_with_teacher, stop_token_with_teacher, decoder_state_with_teacher = decoder(encoder_output,
                                                                                                       attention_fn=attention_factory(
                                                                                                           params),
                                                                                                       speaker_embed=speaker_embedding_output,
                                                                                                       is_training=is_training,
                                                                                                       is_validation=is_validation,
                                                                                                       memory_sequence_length=features.source_length,
                                                                                                       target_sequence_length=labels.target_length,
                                                                                                       target=target,
                                                                                                       teacher_forcing=True,
                                                                                                       apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.decoder == "TransformerDecoder" and not is_training:
                    alignment_with_teacher = tf.transpose(
                        decoder_state_with_teacher.rnn_state.rnn_state[0].alignment_history.stack(),
                        [1, 2, 0])
                else:
                    alignment_with_teacher = tf.transpose(decoder_state_with_teacher[0].alignment_history.stack(),
                                                          [1, 2, 0])

                if params.use_postnet_v2:
                    postnet_v2_mel_output_with_teacher = postnet(mel_output_with_teacher)

                mel_loss_with_teacher = spec_loss(mel_output_with_teacher, labels.mel,
                                                  labels.spec_loss_mask, params.spec_loss_type)
                done_loss_with_teacher = binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                postnet_v2_mel_loss_with_teacher = spec_loss(postnet_v2_mel_output_with_teacher, labels.mel,
                                                             labels.spec_loss_mask, params.spec_loss_type) if params.use_postnet_v2 else 0
                loss_with_teacher = mel_loss_with_teacher + done_loss_with_teacher + regularization_loss + postnet_v2_mel_loss_with_teacher

                eval_metric_ops = self.get_validation_metrics(mel_loss, done_loss, postnet_v2_mel_loss,
                                                              loss_with_teacher,
                                                              mel_loss_with_teacher, done_loss_with_teacher,
                                                              postnet_v2_mel_loss_with_teacher,
                                                              regularization_loss)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MetricsSaver([alignment] + decoder_self_attention_alignment, global_step, mel_output,
                                               labels.mel,
                                               labels.target_length,
                                               features.id,
                                               features.text,
                                               1,
                                               mode, summary_writer,
                                               save_training_time_metrics=params.save_training_time_metrics,
                                               keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                num_self_alignments = len(decoder_self_attention_alignment)
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "mel": mel_output,
                    "mel_postnet": postnet_v2_mel_output if params.use_postnet_v2 else None,
                    "ground_truth_mel": features.mel,
                    "alignment": alignment,
                    "alignment2": decoder_self_attention_alignment[0] if num_self_alignments >= 1 else None,
                    "alignment3": decoder_self_attention_alignment[1] if num_self_alignments >= 2 else None,
                    "alignment4": decoder_self_attention_alignment[2] if num_self_alignments >= 3 else None,
                    "alignment5": decoder_self_attention_alignment[3] if num_self_alignments >= 4 else None,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(ExtendedTacotronV1Model, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def learning_rate_decay(init_rate, global_step, step_factor):
        warmup_steps = 4000.0
        step = tf.to_float(global_step * step_factor + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, mel_loss, done_loss, learning_rate, postnet_v2_mel_loss, l2_regularization_loss):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if mel_loss is not None:
            tf.summary.scalar("mel_loss", mel_loss)
            tf.summary.scalar("mel_loss_with_teacher", mel_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        if postnet_v2_mel_loss is not None:
            tf.summary.scalar("postnet_v2_mel_loss", postnet_v2_mel_loss)
            tf.summary.scalar("postnet_v2_mel_loss_with_teacher", postnet_v2_mel_loss)
        if l2_regularization_loss is not None:
            tf.summary.scalar("l2_regularization_loss", l2_regularization_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(mel_loss, done_loss, postnet_v2_mel_loss, loss_with_teacher, mel_loss_with_teacher,
                               done_loss_with_teacher, postnet_v2_mel_loss_with_teacher, l2_regularization_loss):
        metrics = {}
        if mel_loss is not None:
            metrics["mel_loss"] = tf.metrics.mean(mel_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if postnet_v2_mel_loss is not None:
            metrics["postnet_v2_mel_loss"] = tf.metrics.mean(postnet_v2_mel_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if mel_loss_with_teacher is not None:
            metrics["mel_loss_with_teacher"] = tf.metrics.mean(mel_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        if postnet_v2_mel_loss_with_teacher is not None:
            metrics["postnet_v2_mel_loss_with_teacher"] = tf.metrics.mean(postnet_v2_mel_loss_with_teacher)
        if l2_regularization_loss is not None:
            metrics["l2_regularization_loss"] = tf.metrics.mean(l2_regularization_loss)
        return metrics


class DualSourceSelfAttentionTacotronModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset)

            encoder = encoder_factory(params, is_training)

            assert params.decoder in ["DualSourceDecoder", "DualSourceTransformerDecoder"]
            decoder = decoder_factory(params)

            ## make sure that only one of (external_speaker_embedding, speaker_embedding) has been chosen
            assert not (params.use_speaker_embedding and params.use_external_speaker_embedding)
            
            if params.use_speaker_embedding:
                speaker_embedding = Embedding(params.num_speakers,
                                              embedding_dim=params.speaker_embedding_dim,
                                              index_offset=params.speaker_embedding_offset)
            elif params.use_external_speaker_embedding:
                speaker_embedding = ExternalEmbedding(params.embedding_file, params.num_speakers,
                                                      embedding_dim=params.speaker_embedding_dim,
                                                      index_offset=params.speaker_embedding_offset)

            ## resize speaker embedding with a projection layer
            if params.speaker_embedding_projection_out_dim > -1:
                def _compose(f, g):
                    return lambda arg, *args, **kwargs: f(g(arg, *args, **kwargs))
                resize = tf.layers.Dense(params.speaker_embedding_projection_out_dim, activation=tf.nn.relu)
                speaker_embedding = _compose(resize, speaker_embedding)

            ## language (dialect) embedding
            if params.use_language_embedding:
                language_embedding = ExternalEmbedding(params.language_embedding_file, params.num_speakers,
                                                       embedding_dim=params.language_embedding_dim,
                                                       index_offset=params.speaker_embedding_offset)

            # resize language embedding with a projection layer
            if params.language_embedding_projection_out_dim > -1:  
                def _compose(f, g):
                    return lambda arg, *args, **kwargs: f(g(arg, *args, **kwargs))
                resize = tf.layers.Dense(params.language_embedding_projection_out_dim, activation=tf.nn.relu)
                language_embedding = _compose(resize, language_embedding)

            ## channel label
            if params.channel_id_to_postnet:
                channel_code = ExternalEmbedding(params.channel_id_file, params.num_speakers, embedding_dim=params.channel_id_dim, index_offset=params.speaker_embedding_offset)

            target = labels.mel if (is_training or is_validation) else features.mel

            ## choose a speaker ID to synthesize as
            x = params.speaker_for_synthesis
            if x > -1:
                speaker_embedding_output = speaker_embedding(x)
            else:  ## default is to just use the speaker ID associated with the test utterance
                speaker_embedding_output = speaker_embedding(
                    features.speaker_id) if params.use_speaker_embedding or params.use_external_speaker_embedding else None


            if x > -1:  ## -1 is default (just use the speaker ID associated with the test utterance)
                language_embedding_output = language_embedding(x)
            else:
                language_embedding_output = language_embedding(
                    features.speaker_id) if params.use_language_embedding else None

            channel_code_output = channel_code(features.speaker_id) if params.channel_id_to_postnet else None

            ## get phone/letter embeddings
            embedding_output = embedding(features.source) # phone/letter embedding 
            # add language embedding as bias along the time axis to embedding_output 
            if params.language_embedd_to_input:
                language_embedd_input_projection_layer = tf.layers.Dense(params.embedding_dim)
                language_embedd_input_projected = language_embedd_input_projection_layer(language_embedding_output)
                expand_language_embedding_input = tf.tile(tf.expand_dims(language_embedd_input_projected, axis=1),
                                                          [1, tf.shape(embedding_output)[1], 1])
                embedding_output = embedding_output + expand_language_embedding_input # as bias

            ## pass input embeddings to encoder
            encoder_lstm_output, encoder_self_attention_output, self_attention_alignment = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            ## concatenate encoder outputs with speaker embedding along the time axis
            if params.speaker_embedd_to_decoder:
                expand_speaker_embedding_output = tf.tile(tf.expand_dims(speaker_embedding_output, axis=1),
                                                          [1, tf.shape(encoder_lstm_output)[1], 1])
                encoder_lstm_output = tf.concat((encoder_lstm_output, expand_speaker_embedding_output), axis=-1)
                encoder_self_attention_output = tf.concat(
                    (encoder_self_attention_output, expand_speaker_embedding_output), axis=-1)
                
            # concatenate encoder outputs with language embedding along the time axis
            if params.language_embedd_to_decoder:
                expand_language_embedding_output = tf.tile(tf.expand_dims(language_embedding_output, axis=1),
                                                          [1, tf.shape(encoder_lstm_output)[1], 1])
                encoder_lstm_output = tf.concat((encoder_lstm_output, expand_language_embedding_output), axis=-1)
                encoder_self_attention_output = tf.concat(
                    (encoder_self_attention_output, expand_language_embedding_output), axis=-1)


            attention1_fn, attention2_fn = dual_source_attention_factory(params)
            mel_output, stop_token, decoder_state = decoder((encoder_lstm_output, encoder_self_attention_output),
                                                            attention1_fn=attention1_fn,
                                                            attention2_fn=attention2_fn,
                                                            speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                                                            is_training=is_training,
                                                            is_validation=is_validation or params.use_forced_alignment_mode,
                                                            teacher_forcing=params.use_forced_alignment_mode,
                                                            memory_sequence_length=features.source_length,
                                                            memory2_sequence_length=features.source_length,
                                                            target_sequence_length=labels.target_length if is_training else None,
                                                            target=target,
                                                            apply_dropout_on_inference=params.apply_dropout_on_inference)

            # arrange to (B, T_memory, T_query)
            self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in self_attention_alignment]
            if params.decoder == "DualSourceTransformerDecoder" and not is_training:
                decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                decoder_rnn_state = decoder_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_forced_alignment_mode:
                attention1_fn, attention2_fn = force_alignment_dual_source_attention_factory(params)
                mel_output, stop_token, decoder_state = decoder((encoder_lstm_output, encoder_self_attention_output),
                                                                attention1_fn=attention1_fn,
                                                                attention2_fn=attention2_fn,
                                                                speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                                                                is_training=is_training,
                                                                is_validation=True,
                                                                teacher_forcing=False,
                                                                memory_sequence_length=features.source_length,
                                                                memory2_sequence_length=features.source_length,
                                                                target_sequence_length=labels.target_length if is_training else None,
                                                                target=target,
                                                                teacher_alignments=(
                                                                    tf.transpose(alignment1, perm=[0, 2, 1]),
                                                                    tf.transpose(alignment2, perm=[0, 2, 1])),
                                                                apply_dropout_on_inference=params.apply_dropout_on_inference)
                if params.decoder == "DualSourceTransformerDecoder" and not is_training:
                    alignment1 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[0].stack(),
                                              [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[1].stack(),
                                              [1, 2, 0])
                    decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                        decoder_state.alignments]
                else:
                    alignment1 = tf.transpose(decoder_state[0].alignment_history[0].stack(), [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state[0].alignment_history[1].stack(), [1, 2, 0])
                    decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_postnet_v2:
                postnet = MultiSpeakerPostNet(out_units=params.num_mels,
                                              speaker_embed=speaker_embedding_output,
                                              num_postnet_layers=params.num_postnet_v2_layers,
                                              kernel_size=params.postnet_v2_kernel_size,
                                              out_channels=params.postnet_v2_out_channels,
                                              is_training=is_training,
                                              drop_rate=params.postnet_v2_drop_rate) if params.speaker_embedd_to_postnet \
                    else ChannelEncoderPostNet(out_units=params.num_mels,
                                               channel_code=channel_code_output,
                                               num_postnet_layers=params.num_postnet_v2_layers,
                                               kernel_size=params.postnet_v2_kernel_size,
                                               out_channels=params.postnet_v2_out_channels,
                                               is_training=is_training,
                                               drop_rate=params.postnet_v2_drop_rate) if params.channel_id_to_postnet \
                    else PostNetV2(out_units=params.num_mels,
                                   num_postnet_layers=params.num_postnet_v2_layers,
                                   kernel_size=params.postnet_v2_kernel_size,
                                   out_channels=params.postnet_v2_out_channels,
                                   is_training=is_training,
                                   drop_rate=params.postnet_v2_drop_rate)

                postnet_v2_mel_output = postnet(mel_output)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                mel_loss = spec_loss(mel_output, labels.mel,
                                     labels.spec_loss_mask, params.spec_loss_type)
                done_loss = binary_loss(stop_token, labels.done, labels.binary_loss_mask)

                blacklist = ["embedding", "bias", "batch_normalization", "output_projection_wrapper/kernel",
                             "lstm_cell",
                             "output_and_stop_token_wrapper/dense/", "output_and_stop_token_wrapper/dense_1/",
                             "stop_token_projection/kernel"]
                regularization_loss = l2_regularization_loss(
                    tf.trainable_variables(), params.l2_regularization_weight,
                    blacklist) if params.use_l2_regularization else 0

                postnet_v2_mel_loss = spec_loss(postnet_v2_mel_output, labels.mel,
                                                labels.spec_loss_mask,
                                                params.spec_loss_type) if params.use_postnet_v2 else 0
                loss = mel_loss + done_loss + regularization_loss + postnet_v2_mel_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step,
                    params.learning_rate_step_factor) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, mel_loss, done_loss, lr, postnet_v2_mel_loss, regularization_loss)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MetricsSaver([alignment1, alignment2] + self_attention_alignment, global_step,
                                                   mel_output, labels.mel,
                                                   labels.target_length,
                                                   features.id,
                                                   features.text,
                                                   params.alignment_save_steps,
                                                   mode, summary_writer,
                                                   save_training_time_metrics=params.save_training_time_metrics,
                                                   keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                attention1_fn, attention2_fn = dual_source_attention_factory(params)
                mel_output_with_teacher, stop_token_with_teacher, decoder_state_with_teacher = decoder(
                    (encoder_lstm_output, encoder_self_attention_output),
                    attention1_fn=attention1_fn,
                    attention2_fn=attention2_fn,
                    speaker_embed=speaker_embedding_output if params.speaker_embedd_to_prenet else None,
                    is_training=is_training,
                    is_validation=is_validation,
                    memory_sequence_length=features.source_length,
                    memory2_sequence_length=features.source_length,
                    target_sequence_length=labels.target_length,
                    target=target,
                    teacher_forcing=True,
                    apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.use_postnet_v2:
                    postnet_v2_mel_output_with_teacher = postnet(mel_output_with_teacher)

                mel_loss_with_teacher = spec_loss(mel_output_with_teacher, labels.mel,
                                                  labels.spec_loss_mask, params.spec_loss_type)
                done_loss_with_teacher = binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                postnet_v2_mel_loss_with_teacher = spec_loss(postnet_v2_mel_output_with_teacher, labels.mel,
                                                             labels.spec_loss_mask,
                                                             params.spec_loss_type) if params.use_postnet_v2 else 0
                loss_with_teacher = mel_loss_with_teacher + done_loss_with_teacher + regularization_loss + postnet_v2_mel_loss_with_teacher

                eval_metric_ops = self.get_validation_metrics(mel_loss, done_loss, postnet_v2_mel_loss,
                                                              loss_with_teacher,
                                                              mel_loss_with_teacher, done_loss_with_teacher,
                                                              postnet_v2_mel_loss_with_teacher,
                                                              regularization_loss)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MetricsSaver(
                    [alignment1, alignment2] + self_attention_alignment + decoder_self_attention_alignment, global_step,
                    mel_output, labels.mel,
                    labels.target_length,
                    features.id,
                    features.text,
                    1,
                    mode, summary_writer,
                    save_training_time_metrics=params.save_training_time_metrics,
                    keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                num_self_alignments = len(self_attention_alignment)
                num_decoder_self_alignments = len(decoder_self_attention_alignment)
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "mel": mel_output,
                    "mel_postnet": postnet_v2_mel_output if params.use_postnet_v2 else None,
                    "ground_truth_mel": features.mel,
                    "alignment": alignment1,
                    "alignment2": alignment2,
                    "alignment3": decoder_self_attention_alignment[0] if num_decoder_self_alignments >= 1 else None,
                    "alignment4": decoder_self_attention_alignment[1] if num_decoder_self_alignments >= 2 else None,
                    "alignment5": self_attention_alignment[0] if num_self_alignments >= 1 else None,
                    "alignment6": self_attention_alignment[1] if num_self_alignments >= 2 else None,
                    "alignment7": self_attention_alignment[2] if num_self_alignments >= 3 else None,
                    "alignment8": self_attention_alignment[3] if num_self_alignments >= 4 else None,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(DualSourceSelfAttentionTacotronModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def learning_rate_decay(init_rate, global_step, step_factor):
        warmup_steps = 4000.0
        step = tf.to_float(global_step * step_factor + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, mel_loss, done_loss, learning_rate, postnet_v2_mel_loss, l2_regularization_loss):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if mel_loss is not None:
            tf.summary.scalar("mel_loss", mel_loss)
            tf.summary.scalar("mel_loss_with_teacher", mel_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        if postnet_v2_mel_loss is not None:
            tf.summary.scalar("postnet_v2_mel_loss", postnet_v2_mel_loss)
            tf.summary.scalar("postnet_v2_mel_loss_with_teacher", postnet_v2_mel_loss)
        if l2_regularization_loss is not None:
            tf.summary.scalar("l2_regularization_loss", l2_regularization_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(mel_loss, done_loss, postnet_v2_mel_loss, loss_with_teacher, mel_loss_with_teacher,
                               done_loss_with_teacher, postnet_v2_mel_loss_with_teacher, l2_regularization_loss):
        metrics = {}
        if mel_loss is not None:
            metrics["mel_loss"] = tf.metrics.mean(mel_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if postnet_v2_mel_loss is not None:
            metrics["postnet_v2_mel_loss"] = tf.metrics.mean(postnet_v2_mel_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if mel_loss_with_teacher is not None:
            metrics["mel_loss_with_teacher"] = tf.metrics.mean(mel_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        if postnet_v2_mel_loss_with_teacher is not None:
            metrics["postnet_v2_mel_loss_with_teacher"] = tf.metrics.mean(postnet_v2_mel_loss_with_teacher)
        if l2_regularization_loss is not None:
            metrics["l2_regularization_loss"] = tf.metrics.mean(l2_regularization_loss)
        return metrics


class DualSourceSelfAttentionMgcLf0TacotronModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset)

            encoder = encoder_factory(params, is_training)

            assert params.decoder in ["MgcLf0DualSourceDecoder", "DualSourceMgcLf0TransformerDecoder"]

            decoder = decoder_factory(params)

            if params.use_speaker_embedding:
                speaker_embedding = Embedding(params.num_speakers,
                                              embedding_dim=params.speaker_embedding_dim,
                                              index_offset=params.speaker_embedding_offset)

            target = (labels.mgc, labels.lf0) if (is_training or is_validation) else (features.mgc, features.lf0)

            embedding_output = embedding(features.source)
            encoder_lstm_output, encoder_self_attention_output, self_attention_alignment = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            speaker_embedding_output = speaker_embedding(features.speaker_id) if params.use_speaker_embedding else None

            attention1_fn, attention2_fn = dual_source_attention_factory(params)
            mgc_output, lf0_output, stop_token, decoder_state = decoder(
                (encoder_lstm_output, encoder_self_attention_output),
                attention1_fn=attention1_fn,
                attention2_fn=attention2_fn,
                speaker_embed=speaker_embedding_output,
                is_training=is_training,
                is_validation=is_validation or params.use_forced_alignment_mode,
                teacher_forcing=params.use_forced_alignment_mode,
                memory_sequence_length=features.source_length,
                memory2_sequence_length=features.source_length,
                target=target,
                apply_dropout_on_inference=params.apply_dropout_on_inference)

            # arrange to (B, T_memory, T_query)
            self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in self_attention_alignment]
            if params.decoder == "DualSourceMgcLf0TransformerDecoder" and not is_training:
                decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                decoder_rnn_state = decoder_state[0]
                alignment1 = tf.transpose(decoder_rnn_state.alignment_history[0].stack(), [1, 2, 0])
                alignment2 = tf.transpose(decoder_rnn_state.alignment_history[1].stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_forced_alignment_mode:
                attention1_fn, attention2_fn = force_alignment_dual_source_attention_factory(params)
                mgc_output, lf0_output, stop_token, decoder_state = decoder(
                    (encoder_lstm_output, encoder_self_attention_output),
                    attention1_fn=attention1_fn,
                    attention2_fn=attention2_fn,
                    speaker_embed=speaker_embedding_output,
                    is_training=is_training,
                    is_validation=True,
                    teacher_forcing=False,
                    memory_sequence_length=features.source_length,
                    memory2_sequence_length=features.source_length,
                    target_sequence_length=labels.target_length if is_training else None,
                    target=target,
                    teacher_alignments=(
                        tf.transpose(alignment1, perm=[0, 2, 1]),
                        tf.transpose(alignment2, perm=[0, 2, 1])),
                    apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.decoder == "DualSourceMgcLf0TransformerDecoder" and not is_training:
                    alignment1 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[0].stack(),
                                              [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history[1].stack(),
                                              [1, 2, 0])
                    decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                        decoder_state.alignments]
                else:
                    alignment1 = tf.transpose(decoder_state[0].alignment_history[0].stack(), [1, 2, 0])
                    alignment2 = tf.transpose(decoder_state[0].alignment_history[1].stack(), [1, 2, 0])
                    decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_postnet_v2:
                postnet = PostNetV2(out_units=params.num_mgcs,
                                    num_postnet_layers=params.num_postnet_v2_layers,
                                    kernel_size=params.postnet_v2_kernel_size,
                                    out_channels=params.postnet_v2_out_channels,
                                    is_training=is_training,
                                    drop_rate=params.postnet_v2_drop_rate)

                postnet_v2_mgc_output = postnet(mgc_output)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                mgc_loss = spec_loss(mgc_output, target[0],
                                     labels.spec_loss_mask, params.spec_loss_type)

                lf0_loss = classification_loss(lf0_output, target[1], labels.spec_loss_mask)

                done_loss = binary_loss(stop_token, labels.done, labels.binary_loss_mask)

                postnet_v2_mgc_loss = spec_loss(postnet_v2_mgc_output, target,
                                                labels.spec_loss_mask,
                                                params.spec_loss_type) if params.use_postnet_v2 else 0
                loss = mgc_loss + lf0_loss * params.lf0_loss_factor + done_loss + postnet_v2_mgc_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step,
                    params.learning_rate_step_factor) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, mgc_loss, lf0_loss, done_loss, lr, postnet_v2_mgc_loss)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MgcLf0MetricsSaver([alignment1, alignment2] + self_attention_alignment,
                                                         global_step,
                                                         mgc_output, labels.mgc,
                                                         tf.nn.softmax(lf0_output), labels.lf0,
                                                         labels.target_length,
                                                         features.id,
                                                         features.text,
                                                         params.alignment_save_steps,
                                                         mode, params, summary_writer)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                attention1_fn, attention2_fn = dual_source_attention_factory(params)
                mgc_output_with_teacher, lf0_output_with_teacher, stop_token_with_teacher, decoder_state_with_teacher = decoder(
                    (encoder_lstm_output, encoder_self_attention_output),
                    attention1_fn=attention1_fn,
                    attention2_fn=attention2_fn,
                    speaker_embed=speaker_embedding_output,
                    is_training=is_training,
                    is_validation=is_validation,
                    memory_sequence_length=features.source_length,
                    memory2_sequence_length=features.source_length,
                    target=target,
                    teacher_forcing=True,
                    apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.use_postnet_v2:
                    postnet_v2_mgc_output_with_teacher = postnet(mgc_output_with_teacher)

                mgc_loss_with_teacher = spec_loss(mgc_output_with_teacher, labels.mgc,
                                                  labels.spec_loss_mask, params.spec_loss_type)
                lf0_loss_with_teacher = classification_loss(lf0_output_with_teacher, target[1],
                                                            labels.spec_loss_mask)
                done_loss_with_teacher = binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                postnet_v2_mgc_loss_with_teacher = spec_loss(postnet_v2_mgc_output_with_teacher, labels.mgc,
                                                             labels.spec_loss_mask,
                                                             params.spec_loss_type) if params.use_postnet_v2 else 0
                loss_with_teacher = mgc_loss_with_teacher + lf0_loss_with_teacher * params.lf0_loss_factor + done_loss_with_teacher + postnet_v2_mgc_loss_with_teacher

                eval_metric_ops = self.get_validation_metrics(mgc_loss, lf0_loss, done_loss, postnet_v2_mgc_loss,
                                                              loss_with_teacher, mgc_loss_with_teacher,
                                                              lf0_loss_with_teacher, done_loss_with_teacher,
                                                              postnet_v2_mgc_loss_with_teacher)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MgcLf0MetricsSaver(
                    [alignment1, alignment2] + self_attention_alignment + decoder_self_attention_alignment, global_step,
                    mgc_output, labels.mgc,
                    tf.nn.softmax(lf0_output), labels.lf0,
                    labels.target_length,
                    features.id,
                    features.text,
                    1,
                    mode, params, summary_writer)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                num_self_alignments = len(self_attention_alignment)
                num_decoder_self_alignments = len(decoder_self_attention_alignment)
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "mgc": mgc_output,
                    "mgc_postnet": postnet_v2_mgc_output if params.use_postnet_v2 else None,
                    "lf0": tf.nn.softmax(lf0_output),
                    "ground_truth_mgc": features.mgc,
                    "ground_truth_lf0": features.lf0,
                    "alignment": alignment1,
                    "alignment2": alignment2,
                    "alignment3": decoder_self_attention_alignment[0] if num_decoder_self_alignments >= 1 else None,
                    "alignment4": decoder_self_attention_alignment[1] if num_decoder_self_alignments >= 2 else None,
                    "alignment5": self_attention_alignment[0] if num_self_alignments >= 1 else None,
                    "alignment6": self_attention_alignment[1] if num_self_alignments >= 2 else None,
                    "alignment7": self_attention_alignment[2] if num_self_alignments >= 3 else None,
                    "alignment8": self_attention_alignment[3] if num_self_alignments >= 4 else None,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(DualSourceSelfAttentionMgcLf0TacotronModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def learning_rate_decay(init_rate, global_step, step_factor):
        warmup_steps = 4000.0
        step = tf.to_float(global_step * step_factor + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, mgc_loss, lf0_loss, done_loss, learning_rate, postnet_v2_mgc_loss):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if mgc_loss is not None:
            tf.summary.scalar("mgc_loss", mgc_loss)
            tf.summary.scalar("mgc_loss_with_teacher", mgc_loss)
        if lf0_loss is not None:
            tf.summary.scalar("lf0_loss", lf0_loss)
            tf.summary.scalar("lf0_loss_with_teacher", lf0_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        if postnet_v2_mgc_loss is not None:
            tf.summary.scalar("postnet_v2_mgc_loss", postnet_v2_mgc_loss)
            tf.summary.scalar("postnet_v2_mgc_loss_with_teacher", postnet_v2_mgc_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(mgc_loss, lf0_loss, done_loss, postnet_v2_mgc_loss, loss_with_teacher,
                               mgc_loss_with_teacher, lf0_loss_with_teacher,
                               done_loss_with_teacher, postnet_v2_mgc_loss_with_teacher):
        metrics = {}
        if mgc_loss is not None:
            metrics["mgc_loss"] = tf.metrics.mean(mgc_loss)
        if lf0_loss is not None:
            metrics["lf0_loss"] = tf.metrics.mean(lf0_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if postnet_v2_mgc_loss is not None:
            metrics["postnet_v2_mgc_loss"] = tf.metrics.mean(postnet_v2_mgc_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if mgc_loss_with_teacher is not None:
            metrics["mgc_loss_with_teacher"] = tf.metrics.mean(mgc_loss_with_teacher)
        if lf0_loss_with_teacher is not None:
            metrics["lf0_loss_with_teacher"] = tf.metrics.mean(lf0_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        if postnet_v2_mgc_loss_with_teacher is not None:
            metrics["postnet_v2_mgc_loss_with_teacher"] = tf.metrics.mean(postnet_v2_mgc_loss_with_teacher)
        return metrics


class MgcLf0TacotronModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset)

            encoder = encoder_factory(params, is_training)

            assert params.decoder in ["MgcLf0Decoder"]

            decoder = decoder_factory(params)

            if params.use_speaker_embedding:
                speaker_embedding = Embedding(params.num_speakers,
                                              embedding_dim=params.speaker_embedding_dim,
                                              index_offset=params.speaker_embedding_offset)

            target = (labels.mgc, labels.lf0) if (is_training or is_validation) else (features.mgc, features.lf0)

            embedding_output = embedding(features.source)
            encoder_output = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            speaker_embedding_output = speaker_embedding(features.speaker_id) if params.use_speaker_embedding else None

            mgc_output, lf0_output, stop_token, decoder_state = decoder(encoder_output,
                                                                        attention_fn=attention_factory(params),
                                                                        speaker_embed=speaker_embedding_output,
                                                                        is_training=is_training,
                                                                        is_validation=is_validation or params.use_forced_alignment_mode,
                                                                        teacher_forcing=params.use_forced_alignment_mode,
                                                                        memory_sequence_length=features.source_length,
                                                                        target_sequence_length=labels.target_length if is_training else None,
                                                                        target=target,
                                                                        apply_dropout_on_inference=params.apply_dropout_on_inference)

            # arrange to (B, T_memory, T_query)
            if params.decoder == "TransformerDecoder" and not is_training:
                decoder_rnn_state = decoder_state.rnn_state.rnn_state[0]
                alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
                decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                    decoder_state.alignments]
            else:
                decoder_rnn_state = decoder_state[0]
                alignment = tf.transpose(decoder_rnn_state.alignment_history.stack(), [1, 2, 0])
                decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_forced_alignment_mode:
                mgc_output, lf0_output, stop_token, decoder_state = decoder(
                    encoder_output,
                    attention_fn=force_alignment_attention_factory(params),
                    speaker_embed=speaker_embedding_output,
                    is_training=is_training,
                    is_validation=True,
                    teacher_forcing=False,
                    memory_sequence_length=features.source_length,
                    target_sequence_length=labels.target_length if is_training else None,
                    target=target,
                    teacher_alignments=tf.transpose(alignment, perm=[0, 2, 1]),
                    apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.decoder == "TransformerDecoder" and not is_training:
                    alignment = tf.transpose(decoder_state.rnn_state.rnn_state[0].alignment_history.stack(), [1, 2, 0])
                    decoder_self_attention_alignment = [tf.transpose(a, perm=[0, 2, 1]) for a in
                                                        decoder_state.alignments]
                else:
                    alignment = tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])
                    decoder_self_attention_alignment = []  # ToDo: fill decoder_self_attention_alignment at training time

            if params.use_postnet_v2:
                postnet = PostNetV2(out_units=params.num_mgcs,
                                    num_postnet_layers=params.num_postnet_v2_layers,
                                    kernel_size=params.postnet_v2_kernel_size,
                                    out_channels=params.postnet_v2_out_channels,
                                    is_training=is_training,
                                    drop_rate=params.postnet_v2_drop_rate)

                postnet_v2_mgc_output = postnet(mgc_output)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                mgc_loss = spec_loss(mgc_output, target[0],
                                     labels.spec_loss_mask, params.spec_loss_type)

                lf0_loss = classification_loss(lf0_output, target[1], labels.spec_loss_mask)

                done_loss = binary_loss(stop_token, labels.done, labels.binary_loss_mask)

                postnet_v2_mgc_loss = spec_loss(postnet_v2_mgc_output, target,
                                                labels.spec_loss_mask,
                                                params.spec_loss_type) if params.use_postnet_v2 else 0
                loss = mgc_loss + lf0_loss * params.lf0_loss_factor + done_loss + postnet_v2_mgc_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step,
                    params.learning_rate_step_factor) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, mgc_loss, lf0_loss, done_loss, lr, postnet_v2_mgc_loss)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MgcLf0MetricsSaver([alignment],
                                                         global_step,
                                                         mgc_output, labels.mgc,
                                                         tf.nn.softmax(lf0_output), labels.lf0,
                                                         labels.target_length,
                                                         features.id,
                                                         features.text,
                                                         params.alignment_save_steps,
                                                         mode, params, summary_writer)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                mgc_output_with_teacher, lf0_output_with_teacher, stop_token_with_teacher, decoder_state_with_teacher = decoder(
                    encoder_output,
                    attention_fn=attention_factory(params),
                    speaker_embed=speaker_embedding_output,
                    is_training=is_training,
                    is_validation=is_validation or params.use_forced_alignment_mode,
                    memory_sequence_length=features.source_length,
                    target_sequence_length=labels.target_length if is_training else None,
                    target=target,
                    teacher_forcing=True,
                    apply_dropout_on_inference=params.apply_dropout_on_inference)

                if params.use_postnet_v2:
                    postnet_v2_mgc_output_with_teacher = postnet(mgc_output_with_teacher)

                mgc_loss_with_teacher = spec_loss(mgc_output_with_teacher, labels.mgc,
                                                  labels.spec_loss_mask, params.spec_loss_type)
                lf0_loss_with_teacher = classification_loss(lf0_output_with_teacher, target[1],
                                                            labels.spec_loss_mask)
                done_loss_with_teacher = binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                postnet_v2_mgc_loss_with_teacher = spec_loss(postnet_v2_mgc_output_with_teacher, labels.mgc,
                                                             labels.spec_loss_mask,
                                                             params.spec_loss_type) if params.use_postnet_v2 else 0
                loss_with_teacher = mgc_loss_with_teacher + lf0_loss_with_teacher * params.lf0_loss_factor + done_loss_with_teacher + postnet_v2_mgc_loss_with_teacher

                eval_metric_ops = self.get_validation_metrics(mgc_loss, lf0_loss, done_loss, postnet_v2_mgc_loss,
                                                              loss_with_teacher, mgc_loss_with_teacher,
                                                              lf0_loss_with_teacher, done_loss_with_teacher,
                                                              postnet_v2_mgc_loss_with_teacher,
                                                              None, None)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MgcLf0MetricsSaver(
                    [alignment] + decoder_self_attention_alignment, global_step,
                    mgc_output, labels.mgc,
                    tf.nn.softmax(lf0_output), labels.lf0,
                    labels.target_length,
                    features.id,
                    features.text,
                    1,
                    mode, params, summary_writer)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "mgc": mgc_output,
                    "mgc_postnet": postnet_v2_mgc_output if params.use_postnet_v2 else None,
                    "lf0": tf.nn.softmax(lf0_output),
                    "ground_truth_mgc": features.mgc,
                    "ground_truth_lf0": features.lf0,
                    "alignment": alignment,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(MgcLf0TacotronModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def learning_rate_decay(init_rate, global_step, step_factor):
        warmup_steps = 4000.0
        step = tf.to_float(global_step * step_factor + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, mgc_loss, lf0_loss, done_loss, learning_rate, postnet_v2_mgc_loss):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if mgc_loss is not None:
            tf.summary.scalar("mgc_loss", mgc_loss)
            tf.summary.scalar("mgc_loss_with_teacher", mgc_loss)
        if lf0_loss is not None:
            tf.summary.scalar("lf0_loss", lf0_loss)
            tf.summary.scalar("lf0_loss_with_teacher", lf0_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        if postnet_v2_mgc_loss is not None:
            tf.summary.scalar("postnet_v2_mgc_loss", postnet_v2_mgc_loss)
            tf.summary.scalar("postnet_v2_mgc_loss_with_teacher", postnet_v2_mgc_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(mgc_loss, lf0_loss, done_loss, postnet_v2_mgc_loss, loss_with_teacher,
                               mgc_loss_with_teacher, lf0_loss_with_teacher,
                               done_loss_with_teacher, postnet_v2_mgc_loss_with_teacher):
        metrics = {}
        if mgc_loss is not None:
            metrics["mgc_loss"] = tf.metrics.mean(mgc_loss)
        if lf0_loss is not None:
            metrics["lf0_loss"] = tf.metrics.mean(lf0_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if postnet_v2_mgc_loss is not None:
            metrics["postnet_v2_mgc_loss"] = tf.metrics.mean(postnet_v2_mgc_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if mgc_loss_with_teacher is not None:
            metrics["mgc_loss_with_teacher"] = tf.metrics.mean(mgc_loss_with_teacher)
        if lf0_loss_with_teacher is not None:
            metrics["lf0_loss_with_teacher"] = tf.metrics.mean(lf0_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        if postnet_v2_mgc_loss_with_teacher is not None:
            metrics["postnet_v2_mgc_loss_with_teacher"] = tf.metrics.mean(postnet_v2_mgc_loss_with_teacher)
        return metrics


def encoder_factory(params, is_training):
    if params.encoder == "SelfAttentionCBHGEncoderWithAccentType":
        encoder = SelfAttentionCBHGEncoderWithAccentType(is_training,
                                                         cbhg_out_units=params.cbhg_out_units,
                                                         conv_channels=params.conv_channels,
                                                         max_filter_width=params.max_filter_width,
                                                         projection1_out_channels=params.projection1_out_channels,
                                                         projection2_out_channels=params.projection2_out_channels,
                                                         num_highway=params.num_highway,
                                                         self_attention_out_units=params.self_attention_out_units,
                                                         self_attention_num_heads=params.self_attention_num_heads,
                                                         self_attention_num_hop=params.self_attention_num_hop,
                                                         self_attention_transformer_num_conv_layers=params.self_attention_transformer_num_conv_layers,
                                                         self_attention_transformer_kernel_size=params.self_attention_transformer_kernel_size,
                                                         prenet_out_units=params.encoder_prenet_out_units_if_accent,
                                                         accent_type_prenet_out_units=params.accent_type_prenet_out_units,
                                                         drop_rate=params.encoder_prenet_drop_rate,
                                                         zoneout_factor_cell=params.zoneout_factor_cell,
                                                         zoneout_factor_output=params.zoneout_factor_output,
                                                         self_attention_drop_rate=params.self_attention_drop_rate)
    elif params.encoder == "SelfAttentionCBHGEncoder":
        encoder = SelfAttentionCBHGEncoder(is_training,
                                           cbhg_out_units=params.cbhg_out_units,
                                           conv_channels=params.conv_channels,
                                           max_filter_width=params.max_filter_width,
                                           projection1_out_channels=params.projection1_out_channels,
                                           projection2_out_channels=params.projection2_out_channels,
                                           num_highway=params.num_highway,
                                           self_attention_out_units=params.self_attention_out_units,
                                           self_attention_num_heads=params.self_attention_num_heads,
                                           self_attention_num_hop=params.self_attention_num_hop,
                                           self_attention_transformer_num_conv_layers=params.self_attention_transformer_num_conv_layers,
                                           self_attention_transformer_kernel_size=params.self_attention_transformer_kernel_size,
                                           prenet_out_units=params.encoder_prenet_out_units,
                                           drop_rate=params.encoder_prenet_drop_rate,
                                           zoneout_factor_cell=params.zoneout_factor_cell,
                                           zoneout_factor_output=params.zoneout_factor_output,
                                           self_attention_drop_rate=params.self_attention_drop_rate)
    elif params.use_accent_type and params.encoder == "EncoderV1WithAccentType":
        encoder = EncoderV1WithAccentType(is_training,
                                          cbhg_out_units=params.cbhg_out_units,
                                          conv_channels=params.conv_channels,
                                          max_filter_width=params.max_filter_width,
                                          projection1_out_channels=params.projection1_out_channels,
                                          projection2_out_channels=params.projection2_out_channels,
                                          num_highway=params.num_highway,
                                          prenet_out_units=params.encoder_prenet_out_units_if_accent,
                                          accent_type_prenet_out_units=params.accent_type_prenet_out_units,
                                          drop_rate=params.encoder_prenet_drop_rate,
                                          use_zoneout=params.use_zoneout_at_encoder,
                                          zoneout_factor_cell=params.zoneout_factor_cell,
                                          zoneout_factor_output=params.zoneout_factor_output)
    elif not params.use_accent_type and params.encoder == "ZoneoutEncoderV1":
        encoder = ZoneoutEncoderV1(is_training,
                                   cbhg_out_units=params.cbhg_out_units,
                                   conv_channels=params.conv_channels,
                                   max_filter_width=params.max_filter_width,
                                   projection1_out_channels=params.projection1_out_channels,
                                   projection2_out_channels=params.projection2_out_channels,
                                   num_highway=params.num_highway,
                                   prenet_out_units=params.encoder_prenet_out_units,
                                   drop_rate=params.encoder_prenet_drop_rate,
                                   use_zoneout=params.use_zoneout_at_encoder,
                                   zoneout_factor_cell=params.zoneout_factor_cell,
                                   zoneout_factor_output=params.zoneout_factor_output)
    elif params.encoder == "EncoderV2":
        encoder = EncoderV2(num_conv_layers=params.encoder_v2_num_conv_layers,
                            kernel_size=params.encoder_v2_kernel_size,
                            out_units=params.encoder_v2_out_units,
                            drop_rate=params.encoder_v2_drop_rate,
                            zoneout_factor_cell=params.zoneout_factor_cell,
                            zoneout_factor_output=params.zoneout_factor_output,
                            is_training=is_training)
    else:
        raise ValueError(f"Unknown encoder: {params.encoder}")
    return encoder


def decoder_factory(params):
    if params.decoder == "ExtendedDecoder":
        decoder = ExtendedDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                  drop_rate=params.decoder_prenet_drop_rate,
                                  attention_out_units=params.attention_out_units,
                                  decoder_version=params.decoder_version,
                                  decoder_out_units=params.decoder_out_units,
                                  num_mels=params.num_mels,
                                  outputs_per_step=params.outputs_per_step,
                                  max_iters=params.max_iters,
                                  n_feed_frame=params.n_feed_frame,
                                  zoneout_factor_cell=params.zoneout_factor_cell,
                                  zoneout_factor_output=params.zoneout_factor_output)
    elif params.decoder == "TransformerDecoder":
        decoder = TransformerDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                     drop_rate=params.decoder_prenet_drop_rate,
                                     attention_out_units=params.attention_out_units,
                                     decoder_version=params.decoder_version,
                                     decoder_out_units=params.decoder_out_units,
                                     num_mels=params.num_mels,
                                     outputs_per_step=params.outputs_per_step,
                                     max_iters=params.max_iters,
                                     n_feed_frame=params.n_feed_frame,
                                     zoneout_factor_cell=params.zoneout_factor_cell,
                                     zoneout_factor_output=params.zoneout_factor_output,
                                     self_attention_out_units=params.decoder_self_attention_out_units,
                                     self_attention_num_heads=params.decoder_self_attention_num_heads,
                                     self_attention_num_hop=params.decoder_self_attention_num_hop,
                                     self_attention_drop_rate=params.decoder_self_attention_drop_rate)
    elif params.decoder == "DualSourceDecoder":
        decoder = DualSourceDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                    drop_rate=params.decoder_prenet_drop_rate,
                                    attention_rnn_out_units=params.attention_out_units,
                                    decoder_version=params.decoder_version,
                                    decoder_out_units=params.decoder_out_units,
                                    num_mels=params.num_mels,
                                    outputs_per_step=params.outputs_per_step,
                                    max_iters=params.max_iters,
                                    n_feed_frame=params.n_feed_frame,
                                    zoneout_factor_cell=params.zoneout_factor_cell,
                                    zoneout_factor_output=params.zoneout_factor_output)
    elif params.decoder == "DualSourceTransformerDecoder":
        decoder = DualSourceTransformerDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                               drop_rate=params.decoder_prenet_drop_rate,
                                               attention_rnn_out_units=params.attention_out_units,
                                               decoder_version=params.decoder_version,
                                               decoder_out_units=params.decoder_out_units,
                                               num_mels=params.num_mels,
                                               outputs_per_step=params.outputs_per_step,
                                               max_iters=params.max_iters,
                                               n_feed_frame=params.n_feed_frame,
                                               zoneout_factor_cell=params.zoneout_factor_cell,
                                               zoneout_factor_output=params.zoneout_factor_output,
                                               self_attention_out_units=params.decoder_self_attention_out_units,
                                               self_attention_num_heads=params.decoder_self_attention_num_heads,
                                               self_attention_num_hop=params.decoder_self_attention_num_hop,
                                               self_attention_drop_rate=params.decoder_self_attention_drop_rate)
    elif params.decoder == "MgcLf0Decoder":
        decoder = MgcLf0Decoder(prenet_out_units=params.decoder_prenet_out_units,
                                drop_rate=params.decoder_prenet_drop_rate,
                                attention_rnn_out_units=params.attention_out_units,
                                decoder_version=params.decoder_version,
                                decoder_out_units=params.decoder_out_units,
                                num_mgcs=params.num_mgcs,
                                num_lf0s=params.num_lf0s,
                                outputs_per_step=params.outputs_per_step,
                                max_iters=params.max_iters,
                                n_feed_frame=params.n_feed_frame,
                                zoneout_factor_cell=params.zoneout_factor_cell,
                                zoneout_factor_output=params.zoneout_factor_output)
    elif params.decoder == "MgcLf0DualSourceDecoder":
        decoder = MgcLf0DualSourceDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                          drop_rate=params.decoder_prenet_drop_rate,
                                          attention_rnn_out_units=params.attention_out_units,
                                          decoder_version=params.decoder_version,
                                          decoder_out_units=params.decoder_out_units,
                                          num_mgcs=params.num_mgcs,
                                          num_lf0s=params.num_lf0s,
                                          outputs_per_step=params.outputs_per_step,
                                          max_iters=params.max_iters,
                                          n_feed_frame=params.n_feed_frame,
                                          zoneout_factor_cell=params.zoneout_factor_cell,
                                          zoneout_factor_output=params.zoneout_factor_output)
    elif params.decoder == "DualSourceMgcLf0TransformerDecoder":
        decoder = DualSourceMgcLf0TransformerDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                                     drop_rate=params.decoder_prenet_drop_rate,
                                                     attention_rnn_out_units=params.attention_out_units,
                                                     decoder_version=params.decoder_version,
                                                     decoder_out_units=params.decoder_out_units,
                                                     num_mgcs=params.num_mgcs,
                                                     num_lf0s=params.num_lf0s,
                                                     outputs_per_step=params.outputs_per_step,
                                                     max_iters=params.max_iters,
                                                     n_feed_frame=params.n_feed_frame,
                                                     zoneout_factor_cell=params.zoneout_factor_cell,
                                                     zoneout_factor_output=params.zoneout_factor_output,
                                                     self_attention_out_units=params.decoder_self_attention_out_units,
                                                     self_attention_num_heads=params.decoder_self_attention_num_heads,
                                                     self_attention_num_hop=params.decoder_self_attention_num_hop,
                                                     self_attention_drop_rate=params.decoder_self_attention_drop_rate)
    else:
        raise ValueError(f"Unknown decoder: {params.decoder}")
    return decoder


def tacotron_model_factory(hparams, model_dir, run_config, warm_start_from=None):
    if hparams.tacotron_model == "MgcLf0TacotronModel":
        model = MgcLf0TacotronModel(hparams, model_dir, config=run_config,
                                    warm_start_from=warm_start_from)
    elif hparams.tacotron_model == "DualSourceSelfAttentionMgcLf0TacotronModel":
        model = DualSourceSelfAttentionMgcLf0TacotronModel(hparams, model_dir,
                                                           config=run_config,
                                                           warm_start_from=warm_start_from)
    elif hparams.tacotron_model == "DualSourceSelfAttentionTacotronModel":
        model = DualSourceSelfAttentionTacotronModel(hparams, model_dir,
                                                     config=run_config,
                                                     warm_start_from=warm_start_from)
    elif hparams.tacotron_model == "ExtendedTacotronV1Model":
        model = ExtendedTacotronV1Model(hparams, model_dir,
                                        config=run_config,
                                        warm_start_from=warm_start_from)
    else:
        raise ValueError(f"Unknown Tacotron model: {hparams.tacotron_model}")
    return model
