# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""
Preprocess VCTK dataset
usage: preprocess_vctk.py [options] <in_dir> <out_dir>

options:
    --hparams=<parmas>                  Ad-hoc replacement of hyper parameters. [default: ].
    --version=<version>                 Version number of VCTK.
    --hparam-json-file=<path>           JSON file contains hyper parameters.
    --source-only                       Process source only.
    --target-only                       Process target only.
    -h, --help                          Show help message.

"""

import csv
import os
import numpy as np
import json
from pyspark import SparkContext
from docopt import docopt
from hparams import hparams, hparams_debug_string

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    version = args["--version"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]

    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            hparams_json = "".join(f.readlines())
            hparams.parse_json(hparams_json)

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

    if version == "0.8":
        from preprocess.vctk import VCTK
        instance = VCTK(in_dir, out_dir, hparams)
    elif version == "0.91":
        from preprocess.vctk_v091 import VCTK
        instance = VCTK(in_dir, out_dir, hparams)
    else:
        raise ValueError("Supported version of VCTK: 0.8, 0.91")

    sc = SparkContext()

    record_rdd = sc.parallelize(instance.list_files())

    if process_source:
        keys = instance.process_sources(record_rdd).collect()

    if process_target:
        target_rdd = instance.process_targets(record_rdd)
        keys = target_rdd.keys()
        average = target_rdd.average()
        stddev = np.sqrt(target_rdd.moment2() - np.square(average))
        min_db = target_rdd.min()

        with open(os.path.join(out_dir, 'hparams.json'), 'w') as f:
            hparams_obj = {
                "num_mels": hparams.num_mels,
                "num_freq": hparams.num_freq,
                "sample_rate": hparams.sample_rate,
                "frame_length_ms": hparams.frame_length_ms,
                "frame_shift_ms": hparams.frame_shift_ms,
                "average_mel_level_db": average.tolist(),
                "stddev_mel_level_db": stddev.tolist(),
                "min_mel_level_db": min_db.tolist(),
            }
            print(json.dumps(hparams_obj, indent=4))
            json.dump(hparams_obj, f, indent=4)

    with open(os.path.join(out_dir, 'list.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in keys:
            writer.writerow([path])
