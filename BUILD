# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        "@tacotron2//:tacotron2",
    ],
)

py_binary(
    name = "predict_mel",
    srcs = [
        "predict_mel.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        "@tacotron2//:tacotron2",
    ],
)

py_library(
    name = "modules",
    srcs = glob(
        ["modules/*.py"],
        exclude = ["**/*_test.py"],
    ),
    srcs_version = "PY3ONLY",
    deps = [
        "@tacotron2//:tacotron2",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "preprocess",
    srcs = glob(
        ["preprocess/*.py"],
        exclude = ["**/*_test.py"],
    ),
    srcs_version = "PY3ONLY",
    deps = [
         ":utils",
    ],
)

py_library(
    name = "utils",
    srcs = glob(
        ["utils/*.py"],
        exclude = ["**/*_test.py"],
    ),
    srcs_version = "PY3ONLY",
    deps = [],
)

py_test(
    name = "transformer_test",
    srcs = ["modules/transformer_test.py"],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":modules",
        "@tacotron2//:tacotron2",
    ],
)

py_binary(
    name = "preprocess_vctk",
    srcs = [
        "preprocess_vctk.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":preprocess",
    ],
)

py_binary(
    name = "preprocess_ljspeech",
    srcs = [
        "preprocess_ljspeech.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":preprocess",
    ],
)

py_binary(
    name = "preprocess_ljspeech_wavenet",
    srcs = [
        "preprocess_ljspeech_wavenet.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        ":preprocess",
    ],
)