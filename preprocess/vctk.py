# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import os
from collections import namedtuple
import tensorflow as tf
import numpy as np
from pyspark import RDD, StorageLevel
from utils.tfrecord import write_tfrecord, int64_feature, bytes_feature
from utils.audio import Audio
from preprocess.cleaners import basic_cleaners
from preprocess.text import text_to_sequence


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


def write_preprocessed_source_data(_id: int, key: str, source: np.ndarray, text, speaker_id, age, gender,
                                   filename: str):
    raw_source = source.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'source': bytes_feature([raw_source]),
        'source_length': int64_feature([len(source)]),
        'text': bytes_feature([text.encode('utf-8')]),
        'speaker_id': int64_feature([speaker_id]),
        'age': int64_feature([age]),
        'gender': int64_feature([gender]),
    }))
    write_tfrecord(example, filename)


class SpeakerInfo(namedtuple("SpeakerInfo", ["id", "age", "gender"])):
    pass


class TxtWavRecord(namedtuple("TxtWavRecord", ["id", "key", "txt_path", "wav_path", "speaker_info"])):
    pass


class MelStatistics(namedtuple("MelStatistics", ["id", "key", "max", "min", "sum", "length", "moment2"])):
    pass


class TargetRDD:
    def __init__(self, rdd: RDD):
        self.rdd = rdd

    def keys(self):
        return self.rdd.map(lambda s: s.key).collect()

    def max(self):
        return self.rdd.map(lambda s: s.max).reduce(lambda a, b: np.maximum(a, b))

    def min(self):
        return self.rdd.map(lambda s: s.min).reduce(lambda a, b: np.minimum(a, b))

    def average(self):
        total_value = self.rdd.map(lambda s: s.sum).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda s: s.length).reduce(lambda a, b: a + b)
        return total_value / total_length

    def moment2(self):
        total_value = self.rdd.map(lambda s: s.moment2).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda s: s.length).reduce(lambda a, b: a + b)
        return total_value / total_length


class VCTK:

    def __init__(self, in_dir, out_dir, hparams, speaker_info_filename='speaker-info.txt'):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.speaker_info_filename = speaker_info_filename
        self.audio = Audio(hparams)

    def list_files(self):
        def wav_files(speaker_info: SpeakerInfo):
            wav_dir = os.path.join(self.in_dir, f"wav48/p{speaker_info.id}")
            return [os.path.join(wav_dir, wav_file) for wav_file in sorted(os.listdir(wav_dir)) if
                    wav_file.endswith('.wav')]

        def text_files(speaker_info: SpeakerInfo):
            txt_dir = os.path.join(self.in_dir, f"txt/p{speaker_info.id}")
            return [os.path.join(txt_dir, txt_file) for txt_file in sorted(os.listdir(txt_dir)) if
                    txt_file.endswith('.txt')]

        def text_and_wav_records(file_pairs, speaker_info):
            def create_record(txt_f, wav_f, speaker_info):
                key1 = os.path.basename(wav_f).strip('.wav')
                key2 = os.path.basename(txt_f).strip('.txt')
                assert key1 == key2
                return TxtWavRecord(0, key1, txt_f, wav_f, speaker_info)

            return [create_record(txt_f, wav_f, speaker_info) for txt_f, wav_f in file_pairs]

        records = sum(
            [text_and_wav_records(zip(text_files(si), wav_files(si)), si) for si in self._load_speaker_info()], [])
        return [TxtWavRecord(i, r.key, r.txt_path, r.wav_path, r.speaker_info) for i, r in enumerate(records)]

    def process_sources(self, rdd: RDD):
        return rdd.map(self._process_txt)

    def process_targets(self, rdd: RDD):
        return TargetRDD(rdd.map(self._process_wav).persist(StorageLevel.MEMORY_AND_DISK))

    def _load_speaker_info(self):
        with open(os.path.join(self.in_dir, self.speaker_info_filename), mode='r', encoding='utf8') as f:
            for l in f.readlines()[1:]:
                si = l.split()
                gender = 0 if si[2] == 'F' else 1
                if str(si[0]) != "315":  # FixMe: Why 315 is missing?
                    yield SpeakerInfo(int(si[0]), int(si[1]), gender)

    def _process_wav(self, record: TxtWavRecord):
        wav = self.audio.load_wav(record.wav_path)
        wav = self.audio.trim(wav)
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32).T
        file_path = os.path.join(self.out_dir, f"{record.key}.target.tfrecord")
        write_preprocessed_target_data(record.id, record.key, mel_spectrogram, file_path)
        return MelStatistics(id=record.id,
                             key=record.key,
                             min=np.min(mel_spectrogram, axis=0),
                             max=np.max(mel_spectrogram, axis=0),
                             sum=np.sum(mel_spectrogram, axis=0),
                             length=len(mel_spectrogram),
                             moment2=np.sum(np.square(mel_spectrogram), axis=0))

    def _process_txt(self, record: TxtWavRecord):
        with open(os.path.join(self.in_dir, record.txt_path), mode='r', encoding='utf8') as f:
            txt = f.readline().rstrip("\n")
            sequence, clean_text = text_to_sequence(txt, basic_cleaners)
            source = np.array(sequence, dtype=np.int64)
            file_path = os.path.join(self.out_dir, f"{record.key}.source.tfrecord")
            write_preprocessed_source_data(record.id, record.key, source, clean_text, record.speaker_info.id,
                                           record.speaker_info.age,
                                           record.speaker_info.gender, file_path)
            return record.key
