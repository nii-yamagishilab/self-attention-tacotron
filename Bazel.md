Bazel is a cutting-edge build tool but is still in beta project. 
Bazel is frequently updated and sometimes gives breaking changes.
Besides Bazel is not supported in some environment, for example NFS.

This document describes how to run commands without Bazel.

# Running commands without Bazel

Commands defined with `py_binary` listed in the [BUILD](./BUILD) file can be run directly from python command.
In this case you should resolve dependencies for that command by yourself in place of Bazel.

The most commands depends on `@tacotron2`. Which version of tacotron2 it depends on is described in [WORKSPACE](./WORKSPACE).

After cloning tacotron2, export PYTHONPATH.
```
mkdir external
git clone git@github.com:nii-yamagishilab/tacotron2.git external/tacotron2

export PYTHONPATH=`pwd`/external:`pwd`/external/tacotron2:$PYTHONPATH
```

Then you can execute python commands that depend on tacotron2.

```
python train.py ...
```
