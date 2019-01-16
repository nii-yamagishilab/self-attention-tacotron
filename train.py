# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""Trainining script for seq2seq text-to-speech synthesis model.
Usage: train.py [options]

Options:
    --source-data-root=<dir>            Directory contains preprocessed source features.
    --target-data-root=<dir>            Directory contains preprocessed target features.
    --checkpoint-dir=<dir>              Directory where to save model checkpoints.
    --selected-list-dir=<dir>           Directory contains test.csv, train.csv, and validation.csv
    --hparams=<parmas>                  Ad-hoc replacement of hyper parameters. [default: ].
    --hparam-json-file=<path>           JSON file contains hyper parameters.
    --multi-gpus                        Use multiple GPUs
    -h, --help                          Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import os
from random import shuffle
from multiprocessing import cpu_count
import logging
from datasets.dataset_factory import dataset_factory, create_from_tfrecord_files
from models.models import tacotron_model_factory
from hparams import hparams, hparams_debug_string


def train_and_evaluate(hparams, model_dir, train_source_files, train_target_files, eval_source_files,
                       eval_target_files, use_multi_gpu):

    interleave_parallelism = get_parallelism(hparams.interleave_cycle_length_cpu_factor,
                                             hparams.interleave_cycle_length_min,
                                             hparams.interleave_cycle_length_max)

    tf.logging.info("Interleave parallelism is %d.", interleave_parallelism)

    def train_input_fn():
        source_and_target_files = list(zip(train_source_files, train_target_files))
        shuffle(source_and_target_files)
        source = [s for s, _ in source_and_target_files]
        target = [t for _, t in source_and_target_files]

        dataset = create_from_tfrecord_files(source, target, hparams,
                                             cycle_length=interleave_parallelism,
                                             buffer_output_elements=hparams.interleave_buffer_output_elements,
                                             prefetch_input_elements=hparams.interleave_prefetch_input_elements)

        zipped = dataset.prepare_and_zip()
        zipped = zipped.cache(hparams.cache_file_name) if hparams.use_cache else zipped
        batched = zipped.filter_by_max_output_length().repeat(count=None).shuffle(
            hparams.suffle_buffer_size).group_by_batch().prefetch(hparams.prefetch_buffer_size)
        return batched.dataset

    def eval_input_fn():
        source_and_target_files = list(zip(eval_source_files, eval_target_files))
        shuffle(source_and_target_files)
        source = tf.data.TFRecordDataset([s for s, _ in source_and_target_files])
        target = tf.data.TFRecordDataset([t for _, t in source_and_target_files])

        dataset = dataset_factory(source, target, hparams)
        zipped = dataset.prepare_and_zip()
        dataset = zipped.filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    distribution = tf.contrib.distribute.MirroredStrategy() if use_multi_gpu else None

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        save_checkpoints_steps=hparams.save_checkpoints_steps,
                                        keep_checkpoint_max=hparams.keep_checkpoint_max,
                                        log_step_count_steps=hparams.log_step_count_steps,
                                        train_distribute=distribution)
    estimator = tacotron_model_factory(hparams, model_dir, run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_parallelism(factor, min_value, max_value):
    return min(max(int(cpu_count() * factor), min_value), max_value)


def load_key_list(filename, in_dir):
    path = os.path.join(in_dir, filename)
    with open(path, mode="r", encoding="utf-8") as f:
        for l in f:
            yield l.rstrip("\n")


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    source_data_root = args["--source-data-root"]
    target_data_root = args["--target-data-root"]
    selected_list_dir = args["--selected-list-dir"]
    use_multi_gpu = args["--multi-gpus"]

    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            json = "".join(f.readlines())
            hparams.parse_json(json)

    hparams.parse(args["--hparams"])

    training_list = list(load_key_list("train.csv", selected_list_dir))
    validation_list = list(load_key_list("validation.csv", selected_list_dir))

    training_source_files = [os.path.join(source_data_root, f"{key}.{hparams.source_file_extension}") for key in
                             training_list]
    training_target_files = [os.path.join(target_data_root, f"{key}.{hparams.target_file_extension}") for key in
                             training_list]
    validation_source_files = [os.path.join(source_data_root, f"{key}.{hparams.source_file_extension}") for key in
                               validation_list]
    validation_target_files = [os.path.join(target_data_root, f"{key}.{hparams.target_file_extension}") for key in
                               validation_list]

    log = logging.getLogger("tensorflow")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(hparams.logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info(hparams_debug_string())

    train_and_evaluate(hparams,
                       checkpoint_dir,
                       training_source_files,
                       training_target_files,
                       validation_source_files,
                       validation_target_files,
                       use_multi_gpu)


if __name__ == '__main__':
    main()
