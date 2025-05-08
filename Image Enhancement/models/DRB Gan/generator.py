import os
import random
from typing import List
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from .layers import drb, enhancing_block 

def build_generator(img_shape=(48, 48, 3), num_drb: int = 3) -> tf.keras.Model:
    inputs = Input(shape=img_shape)

    x = Conv2D(64, 3, padding="same")(inputs)

    for _ in range(num_drb):
        x = drb(x, filters=64)

    x = enhancing_block(x, filters=64)

    enhanced = Conv2D(img_shape[2], 3, padding="same", activation="tanh")(x)

    outputs = inputs + enhanced
    return tf.keras.Model(inputs, outputs, name="Generator")
