# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf


def l2_regularization_loss(weights, scale, blacklist):
    def is_black(name):
        return any([black in name for black in blacklist])

    target_weights = [tf.nn.l2_loss(w) for w in weights if not is_black(w.name)]
    l2_loss = sum(target_weights) * scale
    tf.losses.add_loss(l2_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
    return l2_loss
