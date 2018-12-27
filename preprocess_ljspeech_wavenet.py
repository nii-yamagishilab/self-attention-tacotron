# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""
Preprocess LJSpeech dataset
usage: preprocess_ljspeech_wavenet.py [options] <in_dir> <mel_out_dir> <wav_out_dir>

options:
    --hparams=<parmas>                  Ad-hoc replacement of hyper parameters. [default: ].
    --hparam-json-file=<path>           JSON file contains hyper parameters.
    -h, --help                          Show help message.

"""

from pyspark import SparkContext
from docopt import docopt
from hparams import hparams, hparams_debug_string
from preprocess.ljspeech_wavenet import LJSpeech

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    mel_out_dir = args["<mel_out_dir>"]
    wav_out_dir = args["<wav_out_dir>"]

    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            json = "".join(f.readlines())
            hparams.parse_json(json)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    instance = LJSpeech(in_dir, mel_out_dir, wav_out_dir, hparams)

    sc = SparkContext()

    record_rdd = instance.text_and_path_rdd(sc)

    target_rdd = instance.process_wav(record_rdd)
    target_rdd.collect()
