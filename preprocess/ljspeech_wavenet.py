# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Preprocess for LJSpeech dataset. """

from pyspark import SparkContext, RDD
import numpy as np
import os
from collections import namedtuple
from utils.audio import Audio


class TextAndPath(namedtuple("TextAndPath", ["id", "key", "wav_path", "labels_path", "text"])):
    pass


class LJSpeech:

    def __init__(self, in_dir, mel_out_dir, wav_out_dir, hparams):
        self.in_dir = in_dir
        self.mel_out_dir = mel_out_dir
        self.wav_out_dir = wav_out_dir
        self.audio = Audio(hparams)

    @property
    def record_ids(self):
        return map(lambda v: str(v), range(1, 13101))

    def record_file_path(self, record_id, kind):
        assert kind in ["source", "target"]
        return os.path.join(self.mel_out_dir, f"ljspeech-{kind}-{int(record_id):05d}.tfrecord")

    def text_and_path_rdd(self, sc: SparkContext):
        return sc.parallelize(
            self._extract_all_text_and_path())

    def process_wav(self, rdd: RDD):
        return rdd.mapValues(self._process_wav)

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

    def _process_wav(self, paths: TextAndPath):
        wav = self.audio.load_wav(paths.wav_path)
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32).T
        mel_spectrogram = self.audio.normalize_mel(mel_spectrogram)

        mel_filepath = os.path.join(self.mel_out_dir, f"{paths.key}.mfbsp")
        wav_filepath = os.path.join(self.wav_out_dir, f"{paths.key}.wav")

        mel_spectrogram.tofile(mel_filepath, format="<f4")
        self.audio.save_wav(wav, wav_filepath)
