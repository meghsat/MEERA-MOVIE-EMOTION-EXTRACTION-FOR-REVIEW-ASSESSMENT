
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, AveragePooling2D, UpSampling2D
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
def drb(x, filters: int = 64, num_layers: int = 4):

    residual = x                  
    outputs  = []                 

    for i in range(num_layers):
        inp = x if i == 0 else tf.concat([x] + outputs, axis=-1)
        y   = Conv2D(filters, 3, padding="same")(inp)
        y   = ReLU()(y)
        outputs.append(y)

    fused = Conv2D(filters, 1, padding="same")(tf.concat(outputs, axis=-1))
    return fused + residual       


def enhancing_block(x, filters: int = 64):
    p2  = AveragePooling2D(pool_size=2)(x)
    p4  = AveragePooling2D(pool_size=4)(x)
    p8  = AveragePooling2D(pool_size=8)(x)
    p16 = AveragePooling2D(pool_size=16)(x)
    u2  = UpSampling2D(size=2)(p2)
    u4  = UpSampling2D(size=4)(p4)
    u8  = UpSampling2D(size=8)(p8)
    u16 = UpSampling2D(size=16)(p16)

    fused = tf.concat([x, u2, u4, u8, u16], axis=-1)     
    return Conv2D(filters, 1, padding="same")(fused)      
