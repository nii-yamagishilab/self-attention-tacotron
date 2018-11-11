# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import os
from tacotron2.util.metrics import plot_alignment


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
