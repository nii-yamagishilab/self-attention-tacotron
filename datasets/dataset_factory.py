# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from datasets.vctk.dataset import DatasetSource as VCTKDatasetSource
from datasets.ljspeech.dataset import DatasetSource as LJSpeechDatasetSource
from datasets.blizzard2011.dataset import DatasetSource as Blizzard2011DatasetSource


def dataset_factory(source, target, hparams):
    if hparams.dataset == "vctk.dataset.DatasetSource":
        return VCTKDatasetSource(source, target, hparams)
    elif hparams.dataset == "ljspeech.dataset.DatasetSource":
        return LJSpeechDatasetSource(source, target, hparams)
    elif hparams.dataset == "blizzard2011.dataset.DatasetSource":
        return Blizzard2011DatasetSource(source, target, hparams)
    else:
        raise ValueError("Unkown dataset")


def create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=4,
                               buffer_output_elements=None,
                               prefetch_input_elements=None):
    if hparams.dataset == "vctk.dataset.DatasetSource":
        return VCTKDatasetSource.create_from_tfrecord_files(source_files, target_files, hparams,
                                                            cycle_length=cycle_length,
                                                            buffer_output_elements=buffer_output_elements,
                                                            prefetch_input_elements=prefetch_input_elements)
    elif hparams.dataset == "ljspeech.dataset.DatasetSource":
        return LJSpeechDatasetSource.create_from_tfrecord_files(source_files, target_files, hparams,
                                                                cycle_length=cycle_length,
                                                                buffer_output_elements=buffer_output_elements,
                                                                prefetch_input_elements=prefetch_input_elements)
    elif hparams.dataset == "blizzard2011.dataset.DatasetSource":
        return Blizzard2011DatasetSource.create_from_tfrecord_files(source_files, target_files, hparams,
                                                                    cycle_length=cycle_length,
                                                                    buffer_output_elements=buffer_output_elements,
                                                                    prefetch_input_elements=prefetch_input_elements)
    else:
        raise ValueError("Unkown dataset")
