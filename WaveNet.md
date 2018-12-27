# Waveform inversion with WaveNet


## Prerequisite

We use 2 external tools for waveform inversion.

- https://github.com/nii-yamagishilab/project-CURRENNT-public (extended CURRENT toolkit)
- https://github.com/nii-yamagishilab/project-CURRENNT-scripts (recipes for various models using CURRENT toolkit)

Follow the instruction on the [README](https://github.com/nii-yamagishilab/project-CURRENNT-public/blob/master/CURRENNT_codes/README) and compile CURRENT toolkit.


## Training

We use [project-CURRENNT-scripts/waveform-modeling/project-WaveNet](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/project-WaveNet) to train a WaveNet model.

Read the [README](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/blob/master/waveform-modeling/project-WaveNet/README) and follow the instruction to train a WaveNet model.

There are a few important settings you have to configure.

- Configure `upsampling_rate=200`
- Configure the number utterances for training
- Configure training data directory

Tacotron uses larger frame shift, 12.5ms. In this case upsampling rate is 16000Hz / 1000 * 12.5ms  = 200.
You can configure it [here](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/blob/7e775e0051163d578feb2697829b45c3e9a3fc34/waveform-modeling/project-WaveNet/config.py#L71).

The number of utterances for training is 1000 by default. It is too small so set an appropriate value for the dataset you use.
You can configure it [here](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/blob/7e775e0051163d578feb2697829b45c3e9a3fc34/waveform-modeling/project-WaveNet/config.py#L85).


By default training data location is [project-CURRENNT-scripts/waveform-modeling/DATA/](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/DATA).
You can configure the location [here](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/blob/7e775e0051163d578feb2697829b45c3e9a3fc34/waveform-modeling/project-WaveNet/config.py#L50).
There are two subdirectories, `mfbsp` is for mel-spectrogram, and `wav32k` is for wav files.


You can prepare training data by running the following command in this repository, as an example of LJSpeech.

```
bazel run preprocess_ljspeech_wavenet -- --hparam-json-file=self-attention-tacotron/examples/ljspeech/self-attention-tacotron.json path/to/LJSpeech-1.1 /path/to/mel/output/dir /path/to/wav/output/dir
```

Please set the location `/path/to/mel/output/dir` and `/path/to/wav/output/dir` to the location described above.

You can start pre-processing and training by the following command.

```
./00_run.sh
```


## Prediction

You can find how to predict mel-spectrogram with trained Tacotron model [here](./README.md#prediction).

Please configure the output directory of the mel-spectrogram as an input directory for WaveNet by the following environment variable.

```
export TEMP_WAVEFORM_MODEL_INPUT_DIRS=/path/to/mel-spectrogram/dir
```

Waveform generation can be started by the following command.

```
./01_gen.sh
```