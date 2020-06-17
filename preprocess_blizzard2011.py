# coding: utf-8
"""
Preprocess Blizzard 2011 dataset
usage: preprocess_blizzard2011.py [options] <in_dir> <out_dir>

options:
    --hparams=<parmas>       Hyper parameters. [default: ].
    --source-only            Process source only.
    --target-only            Process target only.
    -h, --help               Show help message.

"""

import csv
import os
import numpy as np
import json
from pyspark import SparkContext
from docopt import docopt
from hparams import hparams, hparams_debug_string
from preprocess.blizzard2011 import Blizzard2011

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    if source_only:
        process_source = True
        process_target = False
    elif target_only:
        process_source = False
        process_target = True
    else:
        process_source = True
        process_target = True

    instance = Blizzard2011(in_dir, out_dir, hparams)

    sc = SparkContext()

    record_rdd = sc.parallelize(instance.list_files())

    if process_source:
        keys = instance.process_sources(record_rdd).collect()

    if process_target:
        target_rdd = instance.process_targets(record_rdd)
        keys = target_rdd.keys()
        average = target_rdd.average()
        stddev = np.sqrt(target_rdd.moment2() - np.square(average))

        with open(os.path.join(out_dir, 'hparams.json'), 'w') as f:
            hparams_obj = {
                "num_mels": hparams.num_mels,
                "num_freq": hparams.num_freq,
                "sample_rate": hparams.sample_rate,
                "frame_length_ms": hparams.frame_length_ms,
                "frame_shift_ms": hparams.frame_shift_ms,
                "mel_fmin": hparams.mel_fmin,
                "mel_fmax": hparams.mel_fmax,
                "average_mel_level_db": list(average),
                "stddev_mel_level_db": list(stddev),
            }
            print(json.dumps(hparams_obj, indent=4))
            json.dump(hparams_obj, f, indent=4)

    with open(os.path.join(out_dir, 'list.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in keys:
            writer.writerow([path])
