"""synthesize waveform
Usage: predict_mel.py [options]

Options:
    --source-data-root=<dir>        Directory contains preprocessed features.
    --target-data-root=<dir>           Directory contains preprocessed features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints.
    --hparams=<parmas>              Ad-hoc replacement of hyper parameters. [default: ].
    --hparam-json-file=<path>       JSON file contains hyper parameters.
    --checkpoint=<path>             Restore model from checkpoint path if given.
    --selected-list-dir=<dir>       Directory contains test.csv, train.csv, and validation.csv
    --output-dir=<path>             Output directory.
    --selected-list-filename=<name> Selected list file name [default: test.csv]
    -h, --help                      Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import os
from collections import namedtuple
from modules.metrics import plot_predictions
from utils.tfrecord import write_prediction_result
from datasets.dataset_factory import dataset_factory
from models.models import tacotron_model_factory
from hparams import hparams, hparams_debug_string


class PredictedMel(
    namedtuple("PredictedMel",
               ["id", "key", "predicted_mel", "predicted_mel_postnet", "predicted_mel_width", "predicted_target_length",
                "ground_truth_mel", "alignment", "alignment2", "alignment3", "alignment4", "alignment5", "alignment6",
                "source", "text", "accent_type"])):
    pass


def predict(hparams,
            model_dir, checkpoint_path, output_dir,
            test_source_files, test_target_files):
    def predict_input_fn():
        source = tf.data.TFRecordDataset(list(test_source_files))
        target = tf.data.TFRecordDataset(list(test_target_files))
        dataset = dataset_factory(source, target, hparams)
        batched = dataset.prepare_and_zip().group_by_batch(
            batch_size=1).merge_target_to_source()
        return batched.dataset

    estimator = tacotron_model_factory(hparams, model_dir, None)

    predictions = map(
        lambda p: PredictedMel(p["id"], p["key"], p["mel"], p.get("mel_postnet"), p["mel"].shape[1], p["mel"].shape[0],
                               p["ground_truth_mel"], p["alignment"], p.get("alignment2"), p.get("alignment3"),
                               p.get("alignment4"), p.get("alignment5"), p.get("alignment6"),
                               p["source"], p["text"], p.get("accent_type")),
        estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path))

    for v in predictions:
        key = v.key.decode('utf-8')
        mel_filename = f"{key}.{hparams.predicted_mel_extension}"
        mel_filepath = os.path.join(output_dir, mel_filename)
        mel = v.predicted_mel_postnet if hparams.use_postnet_v2 else v.predicted_mel
        assert mel.shape[1] == hparams.num_mels
        mel.tofile(mel_filepath, format='<f4')
        text = v.text.decode("utf-8")
        plot_filename = f"{key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        alignments = list(filter(lambda x: x is not None,
                                 [v.alignment, v.alignment2, v.alignment3, v.alignment4, v.alignment5, v.alignment6]))

        plot_predictions(alignments, v.ground_truth_mel, v.predicted_mel, v.predicted_mel_postnet,
                         text, v.key, plot_filepath)
        prediction_filename = f"{key}.tfrecord"
        prediction_filepath = os.path.join(output_dir, prediction_filename)
        write_prediction_result(v.id, key, alignments, mel, v.ground_truth_mel, text, v.source,
                                v.accent_type, prediction_filepath)


def load_key_list(filename, in_dir):
    path = os.path.join(in_dir, filename)
    with open(path, mode="r", encoding="utf-8") as f:
        for l in f:
            yield l.rstrip("\n")


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    source_data_root = args["--source-data-root"]
    target_data_root = args["--target-data-root"]
    selected_list_dir = args["--selected-list-dir"]
    output_dir = args["--output-dir"]
    selected_list_filename = args["--selected-list-filename"] or "test.csv"

    tf.logging.set_verbosity(tf.logging.INFO)

    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            json = "".join(f.readlines())
            hparams.parse_json(json)

    hparams.parse(args["--hparams"])
    tf.logging.info(hparams_debug_string())

    tf.logging.info(f"A selected list file to use: {os.path.join(selected_list_dir, selected_list_filename)}")

    test_list = list(load_key_list(selected_list_filename, selected_list_dir))

    test_source_files = [os.path.join(source_data_root, f"{key}.{hparams.source_file_extension}") for key in
                         test_list]
    test_target_files = [os.path.join(target_data_root, f"{key}.{hparams.target_file_extension}") for key in
                         test_list]

    predict(hparams,
            checkpoint_dir,
            checkpoint_path,
            output_dir,
            test_source_files,
            test_target_files)


if __name__ == '__main__':
    main()
