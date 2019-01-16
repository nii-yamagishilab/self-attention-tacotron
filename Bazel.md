Bazel is a cutting-edge build tool but still in beta status. 
Bazel is frequently updated and sometimes gives breaking changes.
Besides Bazel is not supported in some environment, for example NFS.

This document describes how to run commands defined in this project without Bazel.

# Running commands without Bazel

Commands defined with `py_binary` listed in the [BUILD](./BUILD) file can be run directly from python command.
In this case you should resolve dependencies for that command by yourself in place of Bazel.

You can find the defined commands by looking for `py_binary` rules in [BUILD](./BUILD) file,

or with `bazel query`.
```
bazel query 'kind(py_binary, //...)'

//:train
//:preprocess_vctk
//:preprocess_ljspeech_wavenet
//:preprocess_ljspeech
//:predict_mel
```

You will find these commands are just python executables.
For example, `//:train` command is `train.py` executable.
In my convention, Bazel command name is python file name without `.py` extension.

```
py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        "@tacotron2//:tacotron2",
    ],
)
```


Some commands depends on `@tacotron2`. 
For instance the `//:train` command described above has `@tacotron2` dependency, as defined in `deps` attribute.
Which version of tacotron2 it depends on is described in [WORKSPACE](./WORKSPACE).

You can manually configure tacotron2 dependency by cloning tacotron2 and exporting PYTHONPATH.
```
mkdir external
git clone git@github.com:nii-yamagishilab/tacotron2.git external/tacotron2

export PYTHONPATH=`pwd`/external:`pwd`/external/tacotron2:$PYTHONPATH
```

Then you can execute python commands that depend on tacotron2.

```
cd self-attention-tacotron
python train.py ...
```

Arguments for these command are described in [README](./README.md).