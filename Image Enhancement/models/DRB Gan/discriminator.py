import os
import random
from typing import List
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, LeakyReLU, Flatten, Dense, Input, AveragePooling2D
)

def _conv_block(x, filters, stride=2):
    x = Conv2D(filters, 4, strides=stride, padding="same")(x)
    return LeakyReLU(alpha=0.2)(x)

def build_local_discriminator(patch_shape=(32, 32, 3)) -> tf.keras.Model:

    inp = Input(shape=patch_shape)
    x = _conv_block(inp, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)
    x = _conv_block(x, 512)
    feat = Flatten()(x)               
    logits = Dense(1)(feat)           
    return tf.keras.Model(inp, logits, name="LocalDiscriminator")

def build_global_discriminator(img_shape=(48, 48, 3)) -> tf.keras.Model:

    inp = Input(shape=img_shape)

    pooled = AveragePooling2D(pool_size=2)(inp)   # 48→24
    x = _conv_block(pooled, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)
    x = _conv_block(x, 512)
    feat = Flatten()(x)
    logits = Dense(1)(feat)
    return tf.keras.Model(inp, logits, name="GlobalDiscriminator")
