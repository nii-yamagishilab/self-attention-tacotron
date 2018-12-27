# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Preprocess for LJSpeech dataset. """

from pyspark import SparkContext, RDD, StorageLevel
import tensorflow as tf
import numpy as np
import os
from collections import namedtuple
from utils.tfrecord import bytes_feature, int64_feature, write_tfrecord
from utils.audio import Audio
from preprocess.cleaners import english_cleaners
from preprocess.text import text_to_sequence


class TextAndPath(namedtuple("TextAndPath", ["id", "key", "wav_path", "labels_path", "text"])):
    pass


def write_preprocessed_target_data(_id: int, key: str, mel: np.ndarray, filename: str):
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mel': bytes_feature([raw_mel]),
        'target_length': int64_feature([len(mel)]),
        'mel_width': int64_feature([mel.shape[1]]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_source_data(_id: int, key: str, source: np.ndarray, text, filename: str):
    raw_source = source.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'source': bytes_feature([raw_source]),
        'source_length': int64_feature([len(source)]),
        'text': bytes_feature([text.encode('utf-8')]),
    }))
    write_tfrecord(example, filename)


class MelStatistics(namedtuple("MelStatistics", ["id", "key", "max", "min", "sum", "length", "moment2"])):
    pass


class TargetRDD:
    def __init__(self, rdd: RDD):
        self.rdd = rdd

    def keys(self):
        return self.rdd.map(lambda kv: kv[1].key).collect()

    def max(self):
        return self.rdd.map(lambda kv: kv[1].max).reduce(lambda a, b: np.maximum(a, b))

    def min(self):
        return self.rdd.map(lambda kv: kv[1].min).reduce(lambda a, b: np.minimum(a, b))

    def average(self):
        total_value = self.rdd.map(lambda kv: kv[1].sum).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda kv: kv[1].length).reduce(lambda a, b: a + b)
        return total_value / total_length

    def moment2(self):
        total_value = self.rdd.map(lambda kv: kv[1].moment2).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda kv: kv[1].length).reduce(lambda a, b: a + b)
        return total_value / total_length


class LJSpeech:

    def __init__(self, in_dir, out_dir, hparams):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.audio = Audio(hparams)

    @property
    def record_ids(self):
        return map(lambda v: str(v), range(1, 13101))

    def record_file_path(self, record_id, kind):
        assert kind in ["source", "target"]
        return os.path.join(self.out_dir, f"ljspeech-{kind}-{int(record_id):05d}.tfrecord")

    def text_and_path_rdd(self, sc: SparkContext):
        return sc.parallelize(
            self._extract_all_text_and_path())

    def process_targets(self, rdd: RDD):
        return TargetRDD(rdd.mapValues(self._process_target).persist(StorageLevel.MEMORY_AND_DISK))

    def process_sources(self, rdd: RDD):
        return rdd.mapValues(self._process_source)

    def _extract_text_and_path(self, line, index):
        parts = line.strip().split('|')
        key = parts[0]
        text = parts[2]
        wav_path = os.path.join(self.in_dir, 'wavs', '%s.wav' % key)
        return TextAndPath(index, key, wav_path, None, text)

    def _extract_all_text_and_path(self):
        with open(os.path.join(self.in_dir, 'metadata.csv'), mode='r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                extracted = self._extract_text_and_path(line, index)
                if extracted is not None:
                    yield (index, extracted)

    def _text_to_sequence(self, text):
        sequence, clean_text = text_to_sequence(text, english_cleaners)
        sequence = np.array(sequence, dtype=np.int64)
        return sequence, clean_text

    def _process_target(self, paths: TextAndPath):
        wav = self.audio.load_wav(paths.wav_path)
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32).T
        filename = f"{paths.key}.target.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        write_preprocessed_target_data(paths.id, paths.key, mel_spectrogram, filepath)
        return MelStatistics(id=paths.id,
                             key=paths.key,
                             min=np.min(mel_spectrogram, axis=0),
                             max=np.max(mel_spectrogram, axis=0),
                             sum=np.sum(mel_spectrogram, axis=0),
                             length=len(mel_spectrogram),
                             moment2=np.sum(np.square(mel_spectrogram), axis=0))

    def _process_source(self, paths: TextAndPath):
        sequence, clean_text = self._text_to_sequence(paths.text)
        filename = f"{paths.key}.source.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        write_preprocessed_source_data(paths.id, paths.key, sequence, clean_text, filepath)
        return paths.key
