# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import os
from typing import List
import numpy as np
from tacotron2.util.tfrecord import write_tfrecord, int64_feature, bytes_feature
from tacotron2.util.metrics import plot_alignment


def plot_mels(mel, mel_predicted, mel_input, _id, key, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(3, 1, 1)
    ax.set_title("ground truth")
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(3, 1, 2)
    ax.set_title("output")
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_title("input")
    im = ax.imshow(mel_input[:mel.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}, key: {key}\nglobal step: {global_step}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_predictions(alignments, mel, mel_predicted, text, key, filename):
    from matplotlib import pylab as plt
    num_alignment = len(alignments)
    num_rows = num_alignment + 3
    fig = plt.figure(figsize=(14, num_rows * 3))

    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_rows, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap='jet')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))

    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    ax = fig.add_subplot(num_rows, 1, num_alignment + 1)
    im = ax.imshow(mel.T, origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(num_rows, 1, num_alignment + 2, sharex=ax)
    im = ax.imshow(mel_predicted.T,
                   origin="lower bottom", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)

    fig.suptitle(f"record ID: {key}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_mgc_lf0_spec(mgc, mgc_predicted, spec, spec_predicted, lf0, lf0_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 30))
    ax = fig.add_subplot(6, 1, 1)
    im = ax.imshow(mgc.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(6, 1, 2)
    im = ax.imshow(mgc_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(6, 1, 3)
    im = ax.imshow(spec.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=-40.0, vmax=20.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(6, 1, 4)
    im = ax.imshow(spec_predicted[:spec.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=-40.0, vmax=20.0)
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(6, 1, 5)
    im = ax.imshow(lf0.T, origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(6, 1, 6)
    im = ax.imshow(lf0_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def plot_mgc_lf0(mgc, mgc_predicted, lf0, lf0_predicted, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 20))
    ax = fig.add_subplot(4, 1, 1)
    im = ax.imshow(mgc.T, origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(4, 1, 2)
    im = ax.imshow(mgc_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="magma", vmin=-4.0, vmax=4.0)
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(4, 1, 3)
    im = ax.imshow(lf0.T, origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(4, 1, 4)
    im = ax.imshow(lf0_predicted[:mgc.shape[0], :].T,
                   origin="lower bottom", aspect="auto", cmap="binary", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def write_postnet_v2_result(global_step: int, id: List[int], predicted_mel: List[np.ndarray],
                            ground_truth_mel: List[np.ndarray], input_mel: List[np.ndarray], mel_length: List[int],
                            filename: str):
    batch_size = len(ground_truth_mel)
    raw_predicted_mel = [m.tostring() for m in predicted_mel]
    raw_ground_truth_mel = [m.tostring() for m in ground_truth_mel]
    raw_input_mel = [m.tostring() for m in input_mel]
    mel_width = ground_truth_mel[0].shape[1]
    padded_mel_length = [m.shape[0] for m in ground_truth_mel]
    predicted_mel_length = [m.shape[0] for m in predicted_mel]
    input_mel_length = [m.shape[0] for m in input_mel]
    example = tf.train.Example(features=tf.train.Features(feature={
        'global_step': int64_feature([global_step]),
        'batch_size': int64_feature([batch_size]),
        'id': int64_feature(id),
        'predicted_mel': bytes_feature(raw_predicted_mel),
        'ground_truth_mel': bytes_feature(raw_ground_truth_mel),
        'mel_length': int64_feature(padded_mel_length),
        'mel_length_without_padding': int64_feature(mel_length),
        'predicted_mel_length': int64_feature(predicted_mel_length),
        'input_mel': bytes_feature(raw_input_mel),
        'input_mel_length': int64_feature(input_mel_length),
        'mel_width': int64_feature([mel_width]),
    }))
    write_tfrecord(example, filename)


class PostNetV2MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, global_step_tensor, predicted_mel_tensor, input_mel_tensor, ground_truth_mel_tensor,
                 mel_length_tensor, id_tensor, key_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter):
        self.global_step_tensor = global_step_tensor
        self.predicted_mel_tensor = predicted_mel_tensor
        self.input_mel_tensor = input_mel_tensor
        self.ground_truth_mel_tensor = ground_truth_mel_tensor
        self.mel_length_tensor = mel_length_tensor
        self.id_tensor = id_tensor
        self.key_tensor = key_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, predicted_mel, input_mel, ground_truth_mel, mel_length, ids, keys = run_context.session.run(
                (self.global_step_tensor, self.predicted_mel_tensor, self.input_mel_tensor,
                 self.ground_truth_mel_tensor, self.mel_length_tensor, self.id_tensor, self.key_tensor))
            ids = list(ids)
            id_strings = ",".join([str(i) for i in ids])
            result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
            tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
            write_postnet_v2_result(global_step_value, ids, list(predicted_mel),
                                    list(ground_truth_mel), list(input_mel), list(mel_length),
                                    filename=os.path.join(self.writer.get_logdir(), result_filename))
            if self.mode == tf.estimator.ModeKeys.EVAL:
                for _id, key, pred_mel, gt_mel, in_mel in zip(ids, keys, predicted_mel,
                                                              ground_truth_mel,
                                                              input_mel):
                    output_filename = "{}_result_step{:09d}_{}.png".format(self.mode,
                                                                           global_step_value, _id)
                    plot_mels(gt_mel, pred_mel, in_mel, _id, key, global_step_value,
                              os.path.join(self.writer.get_logdir(), "spec_" + output_filename))


class MgcLf0MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, global_step_tensor, predicted_mgc_tensor, ground_truth_mgc_tensor,
                 predicted_lf0_tensor, ground_truth_lf0_tensor,
                 target_length_tensor, id_tensor,
                 text_tensor, save_steps,
                 mode, hparams, writer: tf.summary.FileWriter):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.predicted_mgc_tensor = predicted_mgc_tensor
        self.ground_truth_mgc_tensor = ground_truth_mgc_tensor
        self.predicted_lf0_tensor = predicted_lf0_tensor
        self.ground_truth_lf0_tensor = ground_truth_lf0_tensor
        self.target_length_tensor = target_length_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            stale_global_step = run_values.results["global_step"]
            if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
                global_step_value, alignments, predicted_mgcs, ground_truth_mgcs, predicted_lf0s, ground_truth_lf0s, target_length, ids, texts = run_context.session.run(
                    (self.global_step_tensor, self.alignment_tensors, self.predicted_mgc_tensor,
                     self.ground_truth_mgc_tensor, self.predicted_lf0_tensor, self.ground_truth_lf0_tensor,
                     self.target_length_tensor, self.id_tensor, self.text_tensor))
                id_strings = ",".join([str(i) for i in ids])
                result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)

                alignments = [[a[i] for a in alignments] for i in range(alignments[0].shape[0])]
                for _id, text, align, pred_mgc, gt_mgc, pred_lf0, gt_lf0 in zip(ids, texts, alignments, predicted_mgcs,
                                                                                ground_truth_mgcs, predicted_lf0s,
                                                                                ground_truth_lf0s):
                    output_filename = "{}_result_step{:09d}_{:d}.png".format(self.mode,
                                                                             global_step_value, _id)
                    plot_alignment(align, text.decode('utf-8'), _id, global_step_value,
                                   os.path.join(self.writer.get_logdir(), "alignment_" + output_filename))
                    plot_mgc_lf0(gt_mgc, pred_mgc, gt_lf0, pred_lf0,
                                 text.decode('utf-8'), _id, global_step_value,
                                 os.path.join(self.writer.get_logdir(), "mgc_lf0_" + output_filename))
