# Copyright (c) 2017 Keith Ito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import librosa
import numpy as np
import scipy


class Audio:
    def __init__(self, hparams):
        self.hparams = hparams
        self._mel_basis = self._build_mel_basis()
        self.average_mel_level_db = np.array(hparams.average_mel_level_db, dtype=np.float32)
        self.stddev_mel_level_db = np.array(hparams.stddev_mel_level_db, dtype=np.float32)

    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(self.hparams.sample_rate, n_fft, n_mels=self.hparams.num_mels)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def save_wav(self, wav, path):
        scipy.io.wavfile.write(path, self.hparams.sample_rate, wav)

    def trim(self, wav):
        unused_trimed, index = librosa.effects.trim(wav, top_db=self.hparams.trim_top_db,
                                                    frame_length=self.hparams.trim_frame_length,
                                                    hop_length=self.hparams.trim_hop_length)
        num_sil_samples = int(
            self.hparams.num_silent_frames * self.hparams.frame_shift_ms * self.hparams.sample_rate / 1000)
        start_idx = max(index[0] - num_sil_samples, 0)
        stop_idx = min(index[1] + num_sil_samples, len(wav))
        trimmed = wav[start_idx:stop_idx]
        return trimmed

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return S

    def normalize_mel(self, S):
        return (S - self.average_mel_level_db) / self.stddev_mel_level_db

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
