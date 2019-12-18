# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import numpy as np
from collections import namedtuple
from abc import abstractmethod
from utils.tfrecord import parse_preprocessed_mel_data, decode_preprocessed_mel_data, \
    PreprocessedMelData


class PreprocessedSourceData(namedtuple("PreprocessedSourceData",
                                        ["id",
                                         "key",
                                         "source",
                                         "source_length",
                                         "speaker_id",
                                         "age",
                                         "gender",
                                         "text",
                                         "phone",
                                         "phone_length",
                                         "phone_txt"])):
    pass


class SourceData(namedtuple("SourceData",
                            ["id",
                             "key",
                             "source",
                             "source_length",
                             "speaker_id",
                             "age",
                             "gender",
                             "text", ])):
    pass


class MelData(
    namedtuple("MelData",
               ["id", "key", "mel", "mel_width", "target_length", "done", "spec_loss_mask", "binary_loss_mask"])):
    pass


class SourceDataForPrediction(namedtuple("SourceDataForPrediction",
                                         ["id",
                                          "key",
                                          "source",
                                          "source_length",
                                          "speaker_id",
                                          "age",
                                          "gender",
                                          "text",
                                          "mel",
                                          "mel_width",
                                          "target_length"])):
    pass


def parse_preprocessed_source_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'source': tf.FixedLenFeature((), tf.string),
        'source_length': tf.FixedLenFeature((), tf.int64),
        'speaker_id': tf.FixedLenFeature((), tf.int64),
        'age': tf.FixedLenFeature((), tf.int64),
        'gender': tf.FixedLenFeature((), tf.int64),
        'text': tf.FixedLenFeature((), tf.string),
        'phone': tf.FixedLenFeature((), tf.string),
        'phone_length': tf.FixedLenFeature((), tf.int64),
        'phone_txt': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_source_data(parsed):
    source = tf.decode_raw(parsed['source'], tf.int64)
    phone = tf.decode_raw(parsed['phone'], tf.int64)
    return PreprocessedSourceData(
        id=parsed["id"],
        key=parsed["key"],
        source=source,
        source_length=parsed["source_length"],
        speaker_id=parsed["speaker_id"],
        age=parsed["age"],
        gender=parsed["gender"],
        text=parsed["text"],
        phone=phone,
        phone_length=parsed["phone_length"],
        phone_txt=parsed["phone_txt"])


class DatasetSource:

    def __init__(self, source, target, hparams):
        self._source = source
        self._target = target
        self._hparams = hparams

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def hparams(self):
        return self._hparams

    @staticmethod
    def create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=4,
                                   buffer_output_elements=None,
                                   prefetch_input_elements=None):
        source = tf.data.Dataset.from_generator(lambda: source_files, tf.string, tf.TensorShape([]))
        target = tf.data.Dataset.from_generator(lambda: target_files, tf.string, tf.TensorShape([]))
        source = source.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length, sloppy=False,
            buffer_output_elements=buffer_output_elements,
            prefetch_input_elements=prefetch_input_elements))
        target = target.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length, sloppy=False,
            buffer_output_elements=buffer_output_elements,
            prefetch_input_elements=prefetch_input_elements))
        return DatasetSource(source, target, hparams)

    def prepare_and_zip(self):
        zipped = tf.data.Dataset.zip(
            (self._prepare_source(self.source, self.hparams), self._prepare_target(self.target, self.hparams)))
        return ZippedDataset(zipped, self.hparams)

    @staticmethod
    def _prepare_source(source, hparams):
        def convert(inputs: PreprocessedSourceData):
            source = inputs.phone if hparams.source == 'phone' else inputs.source
            source_length = inputs.phone_length if hparams.source == 'phone' else inputs.source_length
            text = inputs.phone_txt if hparams.source == 'phone' else inputs.text
            return SourceData(inputs.id, inputs.key, source, source_length, inputs.speaker_id, inputs.age,
                              inputs.gender, text)

        return DatasetSource._decode_source(source).map(lambda inputs: convert(inputs))

    @staticmethod
    def _prepare_target(target, hparams):
        def convert(target: PreprocessedMelData):
            r = hparams.outputs_per_step

            mel_normalized = (target.mel - np.array(hparams.average_mel_level_db, dtype=np.float32)) / np.array(
                hparams.stddev_mel_level_db, dtype=np.float32)

            mel_with_silence = tf.pad(mel_normalized, paddings=[[r, r], [0, 0]],
                                      constant_values=hparams.silence_mel_level_db)

            # +2r for head and tail silence
            target_length = target.target_length + 2 * r
            padded_target_length = (target_length // r + 1) * r

            # spec and mel length must be multiple of outputs_per_step
            def padding_function(t):
                tail_padding = padded_target_length - target_length
                padding_shape = tf.sparse_tensor_to_dense(
                    tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                return lambda: tf.pad(t, paddings=padding_shape, constant_values=hparams.silence_mel_level_db)

            no_padding_condition = tf.equal(tf.to_int64(0), target_length % r)

            mel = tf.cond(no_padding_condition, lambda: mel_with_silence, padding_function(mel_with_silence))

            mel.set_shape((None, hparams.num_mels))

            padded_target_length = tf.cond(no_padding_condition, lambda: target_length, lambda: padded_target_length)

            # done flag
            done = tf.concat([tf.zeros(padded_target_length // r - 1, dtype=tf.float32),
                              tf.ones(1, dtype=tf.float32)], axis=0)

            # loss mask
            spec_loss_mask = tf.ones(shape=padded_target_length, dtype=tf.float32)
            binary_loss_mask = tf.ones(shape=padded_target_length // r, dtype=tf.float32)

            return MelData(target.id, target.key, mel, target.mel_width, padded_target_length, done, spec_loss_mask,
                           binary_loss_mask)

        return DatasetSource._decode_target(target).map(lambda inputs: convert(inputs))

    @staticmethod
    def _decode_source(source):
        return source.map(lambda d: decode_preprocessed_source_data(parse_preprocessed_source_data(d)))

    @staticmethod
    def _decode_target(target):
        return target.map(lambda d: decode_preprocessed_mel_data(parse_preprocessed_mel_data(d)))


class DatasetBase:

    @abstractmethod
    def apply(self, dataset, hparams):
        raise NotImplementedError("apply")

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError("dataset")

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError("hparams")

    def filter(self, predicate):
        return self.apply(self.dataset.filter(predicate), self.hparams)

    def filter_by_max_output_length(self):
        def predicate(s, t: PreprocessedMelData):
            max_output_length = self.hparams.max_iters * self.hparams.outputs_per_step
            return tf.less_equal(t.target_length, max_output_length)

        return self.filter(predicate)

    def shuffle(self, buffer_size):
        return self.apply(self.dataset.shuffle(buffer_size), self.hparams)

    def repeat(self, count=None):
        return self.apply(self.dataset.repeat(count), self.hparams)

    def shuffle_and_repeat(self, buffer_size, count=None):
        dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, count))
        return self.apply(dataset, self.hparams)

    def cache(self, filename):
        return self.apply(self.dataset.cache(filename), self.hparams)


class ZippedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return ZippedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def group_by_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(source, target):
            target_length = tf.minimum(target.target_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                SourceData(
                    id=tf.TensorShape([]),
                    key=tf.TensorShape([]),
                    source=tf.TensorShape([None]),
                    source_length=tf.TensorShape([]),
                    speaker_id=tf.TensorShape([]),
                    age=tf.TensorShape([]),
                    gender=tf.TensorShape([]),
                    text=tf.TensorShape([]),
                ),
                MelData(
                    id=tf.TensorShape([]),
                    key=tf.TensorShape([]),
                    mel=tf.TensorShape([None, self.hparams.num_mels]),
                    mel_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                    done=tf.TensorShape([None]),
                    spec_loss_mask=tf.TensorShape([None]),
                    binary_loss_mask=tf.TensorShape([None]),
                )), padding_values=(
                SourceData(
                    id=tf.to_int64(0),
                    key="",
                    source=tf.to_int64(0),
                    source_length=tf.to_int64(0),
                    speaker_id=tf.to_int64(0),
                    age=tf.to_int64(0),
                    gender=tf.to_int64(-1),
                    text="",
                ),
                MelData(
                    id=tf.to_int64(0),
                    key="",
                    mel=tf.to_float(self.hparams.silence_mel_level_db),
                    mel_width=tf.to_int64(0),
                    target_length=tf.to_int64(0),
                    done=tf.to_float(1),
                    spec_loss_mask=tf.to_float(0),
                    binary_loss_mask=tf.to_float(0),
                )))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        return BatchedDataset(batched, self.hparams)


class BatchedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return BatchedDataset(dataset, self.hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def prefetch(self, buffer_size):
        return self.apply(self.dataset.prefetch(buffer_size), self.hparams)

    def merge_target_to_source(self):
        def convert(s: SourceData, t: MelData):
            return SourceDataForPrediction(
                id=s.id,
                key=s.key,
                source=s.source,
                source_length=s.source_length,
                speaker_id=s.speaker_id,
                age=s.age,
                gender=s.gender,
                text=s.text,
                mel=t.mel,
                mel_width=t.mel_width,
                target_length=t.target_length,
            ), t

        return self.apply(self.dataset.map(convert), self.hparams)
