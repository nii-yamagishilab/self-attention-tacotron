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
    name = "train_postnet",
    srcs = [
        "train_postnet.py",
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

py_binary(
    name = "predict_mgc_lf0",
    srcs = [
        "predict_mgc_lf0.py",
    ],
    srcs_version = "PY3ONLY",
    default_python_version = "PY3",
    deps = [
        "@tacotron2//:tacotron2",
    ],
)

py_binary(
    name = "synthesize",
    srcs = [
        "synthesize.py",
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