# Self-attention Tacotron
An implementation of "Investigation of enhanced Tacotron text-to-speech synthesis systems with self-attention for pitch accent language" https://arxiv.org/abs/1810.11960

**Notice**: Our work in the paper uses a proprietary Japanese speech corpus with manually annotated labels.
Since we cannot provide a exact reproducer in public, this repository replaces dataset related codes with examples for 
publicly available corpus.


## Requirements

Python 3.6 or above is required.

This project uses Bazel as a build tool. 
This project depends on [Tacotron2](https://github.com/nii-yamagishilab/tacotron2) implementation and Bazel automatically resolve the dependency with proper version.

- Python >= 3.6
- Bazel >= 0.18.0

If you are not familiar with Bazel, you can use a python command directly by setting external dependencies by yourself.
See [this document](./Bazel.md) for details.

The following python packages should be installed.

For training and prediction
- tensorflow >= 1.11
- librosa >= 0.6.1
- scipy >= 1.1.1
- matplotlib >= 2.2.2
- docopt >= 0.6.2

For test:
- hypothesis >= 3.59.1

For pre-processing:
- tensorflow >= 1.11
- docopt >= 0.6.2
- pyspark >= 2.3.0
- unidecode >= 1.0.22
- inflect >= 1.0.1


## Preparing data

Pre-process phase generates source and target files in TFRecord format, list containing keys to identify each samples, and hyper parameters.
The source and target files have `.source.tfrecord` and `.target.tfrecord` extension respectively.
The list file is named as `list.csv`. You have to split `list.csv` into `train.csv`, `validation.csv`, and `test.csv`.
Hyper parameters are generated in `hparams.json`. Th important parameters are `average_mel_level_db` and `stddev_mel_level_db`. 
These parameters can be used to normalize spectrogram at training time.

Example configurations for VCTK and LJSpeech can be found in `examples/vctk` and `examples/ljspeech`.


For VCTK, after downloading the corpus, run the following commands.
We recommend to store source and target files separately. You can use `--source-only` and `--target-only` option to do that.

```bash
bazel run preprocess_vctk -- --source-only --hparam-json-file=self-attention-tacotron/examples/vctk/self-attention-tacotron.json /path/to/VCTK-Corpus  /path/to/source/output/dir
bazel run preprocess_vctk -- --target-only --hparam-json-file=self-attention-tacotron/examples/vctk/self-attention-tacotron.json /path/to/VCTK-Corpus  /path/to/target/output/dir
```

For LJSpeech, run the following commands.

```bash
bazel run preprocess_ljspeech -- --source-only --hparam-json-file=self-attention-tacotron/examples/ljspeech/self-attention-tacotron.json /path/to/LJSpeech-1.1  /path/to/source/output/dir
bazel run preprocess_ljspeech -- --target-only --hparam-json-file=self-attention-tacotron/examples/ljspeech/self-attention-tacotron.json /path/to/LJSpeech-1.1  /path/to/target/output/dir
```


## Training

Training script conducts training and validation. 
Validation starts at a certain steps passed. You can control the steps to start validation by setting `save_checkpoints_steps`.
We do not support tensorflow below version 1.11, because behavior of training and validation is different.

`examples` contains configurations for two models: *Self-attention Tacotron* and *baseline Tacotron*.
You can find the configuration files for each model at `self-attention-tacotron.json` and `tacotron.json`.

You can run training by the following command, as an example for Self-attention Tacotron with VCTK dataset.

```bash
bazel run train -- --source-data-root=/path/to/source/output/dir --target-data-root=/path/to/target/output/dir --checkpoint-dir=/path/to/save/checkpoints --selected-list-dir=self-attention-tacotron/examples/vctk --hparam-json-file=self-attention-tacotron/examples/vctk/self-attention-tacotron.json
```

At validation phase, predicted alignments and spectrogram are generated in the checkpoint directory.

You can see summaries like loss value with `tensorboard`. 
Please check `loss_with_teacher` and `mel_loss_with_teacher` for validation metrics.
*xxx_with_teacher* means it is calculated with teacher forcing. 
Since alignment of ground truth and predicted spectrogram does not match normally, reliable metrics are ones with teacher forcing.


## Prediction

You can predict spectrogram with a trained model by the following command, as an example for LJSpeech dataset.

```bash
bazel run predict_mel -- --source-data-root=/path/to/source/output/dir --target-data-root=/path/to/target/output/dir --checkpoint-dir=/path/to/save/checkpoints --output-dir=/path/to/output/results --selected-list-dir=self-attention-tacotron/examples/vctk --hparam-json-file=self-attention-tacotron/examples/ljspeech/self-attention-tacotron.json
```

There are files with `.mfbsp` extension among generated files.
These files are compatible with @TonyWangX 's [WaveNet](https://github.com/nii-yamagishilab/project-CURRENNT-public).
You can find an instruction for waveform inversion with the WaveNet [here](./WaveNet.md).

### Force alignment mode

Force alignment enables to calculate alignment from ground truth spectrogram and use it for predicting spectrogram.

You can use force alignment mode by specifying `use_forced_alignment_mode=True` as *hparams*. 
The following example enables force alignment mode by replacing *hparams* with `--hparams=use_forced_alignment_mode=True`.

```bash
bazel run predict_mel -- --source-data-root=/path/to/source/output/dir --target-data-root=/path/to/target/output/dir --checkpoint-dir=/path/to/save/checkpoints --output-dir=/path/to/output/results --selected-list-dir=self-attention-tacotron/examples/vctk --hparams=use_forced_alignment_mode=True --hparam-json-file=self-attention-tacotron/examples/ljspeech/self-attention-tacotron.json
```


## Running tests

```bash
bazel test //:all --force_python=py3 
```


## ToDo

- [ ] Japanese example with accentual type labels
- [ ] Vocoder parameter examples
- [x] WaveNet instruction


## Licence

BSD 3-Clause License

Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.