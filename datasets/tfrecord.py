# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections.abc import Iterable
from typing import List, Optional


class PreprocessedTargetData(namedtuple("PreprocessedTargetData",
                                        ["id", "key", "spec", "spec_width", "mel", "mel_width", "target_length"])):
    pass


class PreprocessedMelData(namedtuple("PreprocessedMelData",
                                     ["id", "key", "mel", "mel_width", "target_length"])):
    pass


class PreprocessedMgcLf0Data(namedtuple("PreprocessedMgcLf0Data",
                                        ["id", "key", "mgc", "mgc_width", "lf0", "target_length"])):
    pass


class PredictionResult(namedtuple("PredictionResult",
                                  ["id", "key", "mel", "mel_length", "mel_width", "ground_truth_mel",
                                   "ground_truth_mel_length", "text", "source", "source_length"])):
    pass


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def parse_preprocessed_target_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'spec': tf.FixedLenFeature((), tf.string),
        'spec_width': tf.FixedLenFeature((), tf.int64),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'target_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_target_data(parsed):
    spec_width = parsed['spec_width']
    mel_width = parsed['mel_width']
    target_length = parsed['target_length']
    spec = tf.decode_raw(parsed['spec'], tf.float32)
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    return PreprocessedTargetData(
        id=parsed['id'],
        key=parsed['key'],
        spec=tf.reshape(spec, shape=tf.stack([target_length, spec_width], axis=0)),
        spec_width=spec_width,
        mel=tf.reshape(mel, shape=tf.stack([target_length, mel_width], axis=0)),
        mel_width=mel_width,
        target_length=target_length,
    )


def parse_preprocessed_mel_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'target_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_mel_data(parsed):
    mel_width = parsed['mel_width']
    target_length = parsed['target_length']
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    return PreprocessedMelData(
        id=parsed['id'],
        key=parsed['key'],
        mel=tf.reshape(mel, shape=tf.stack([target_length, mel_width], axis=0)),
        mel_width=mel_width,
        target_length=target_length,
    )


def parse_preprocessed_mgc_lf0_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'mgc': tf.FixedLenFeature((), tf.string),
        'mgc_width': tf.FixedLenFeature((), tf.int64),
        'lf0': tf.FixedLenFeature((), tf.string),
        'target_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_mgc_lf0_data(parsed):
    mgc_width = parsed['mgc_width']
    target_length = parsed['target_length']
    mgc = tf.decode_raw(parsed['mgc'], tf.float32)
    lf0 = tf.decode_raw(parsed['lf0'], tf.float32)
    return PreprocessedMgcLf0Data(
        id=parsed['id'],
        key=parsed['key'],
        mgc=tf.reshape(mgc, shape=tf.stack([target_length, mgc_width], axis=0)),
        mgc_width=mgc_width,
        lf0=lf0,
        target_length=target_length,
    )


def write_prediction_result(id_: int, key: str, alignments: List[np.ndarray], mel: np.ndarray,
                            ground_truth_mel: np.ndarray,
                            text: str, source: np.ndarray, accent_type: Optional[np.ndarray], filename: str):
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id_]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mel': bytes_feature([mel.tostring()]),
        'mel_length': int64_feature([mel.shape[0]]),
        'mel_width': int64_feature([mel.shape[1]]),
        'ground_truth_mel': bytes_feature([ground_truth_mel.tostring()]),
        'ground_truth_mel_length': int64_feature([ground_truth_mel.shape[0]]),
        'alignment': bytes_feature([alignment.tostring() for alignment in alignments]),
        'text': bytes_feature([text.encode('utf-8')]),
        'source': bytes_feature([source.tostring()]),
        'source_length': int64_feature([source.shape[0]]),
        'accent_type': bytes_feature([accent_type.tostring()]) if accent_type is not None else bytes_feature([]),
    }))
    write_tfrecord(example, filename)


def write_mgc_lf0_prediction_result(id_: int, key: str, alignments: List[np.ndarray],
                                    mgc: np.ndarray, ground_truth_mgc: np.ndarray,
                                    lf0: np.ndarray, ground_truth_lf0: np.ndarray,
                                    text: str, source: np.ndarray, accent_type: Optional[np.ndarray], filename: str):
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id_]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mgc': bytes_feature([mgc.tostring()]),
        'target_length': int64_feature([mgc.shape[0]]),
        'mgc_width': int64_feature([mgc.shape[1]]),
        'ground_truth_mgc': bytes_feature([ground_truth_mgc.tostring()]),
        'ground_truth_target_length': int64_feature([ground_truth_mgc.shape[0]]),
        'lf0': bytes_feature([lf0.tostring()]),
        'ground_truth_lf0': bytes_feature([ground_truth_lf0.tostring()]),
        'alignment': bytes_feature([alignment.tostring() for alignment in alignments]),
        'text': bytes_feature([text.encode('utf-8')]),
        'source': bytes_feature([source.tostring()]),
        'source_length': int64_feature([source.shape[0]]),
        'accent_type': bytes_feature([accent_type.tostring()]) if accent_type is not None else bytes_feature([]),
    }))
    write_tfrecord(example, filename)


def parse_prediction_result(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_length': tf.FixedLenFeature((), tf.int64),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'ground_truth_mel': tf.FixedLenFeature((), tf.string),
        'ground_truth_mel_length': tf.FixedLenFeature((), tf.int64),
        'alignment': tf.FixedLenFeature((), tf.string),
        'text': tf.FixedLenFeature((), tf.string),
        'source': tf.FixedLenFeature((), tf.string),
        'source_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_prediction_result(parsed):
    source = tf.decode_raw(parsed['source'], tf.int64)
    mel_width = parsed['mel_width']
    mel_length = parsed['mel_length']
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    ground_truth_mel_length = parsed['ground_truth_mel_length']
    ground_truth_mel = tf.decode_raw(parsed['ground_truth_mel'], tf.float32)
    return PredictionResult(
        id=parsed['id'],
        key=parsed['key'],
        mel=tf.reshape(mel, shape=tf.stack([mel_length, mel_width], axis=0)),
        mel_length=mel_length,
        mel_width=mel_width,
        ground_truth_mel=tf.reshape(ground_truth_mel, shape=tf.stack([ground_truth_mel_length, mel_width], axis=0)),
        ground_truth_mel_length=ground_truth_mel_length,
        text=parsed['text'],
        source=source,
        source_length=parsed['source_length'],
    )
