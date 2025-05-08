import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Input

def densely_residual_block(x, filters=64):
    residual = x
    outputs = []

    for i in range(4):
        if i == 0:
            conv = Conv2D(filters, (3, 3), padding='same')(x)
        else:
            concatenated = tf.concat([x] + outputs, axis=-1)
            conv = Conv2D(filters, (3, 3), padding='same')(concatenated)
        conv = ReLU()(conv)
        outputs.append(conv)

    concatenated = tf.concat(outputs, axis=-1)
    fused = Conv2D(filters, (1, 1), padding='same')(concatenated)
    return fused + residual

def enhancing_block(x, filters=64):
    pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
    pool_4 = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))(x)
    pool_8 = tf.keras.layers.AveragePooling2D(pool_size=(16, 16))(x)

    up_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool_1)
    up_2 = tf.keras.layers.UpSampling2D(size=(4, 4))(pool_2)
    up_4 = tf.keras.layers.UpSampling2D(size=(8, 8))(pool_4)
    up_8 = tf.keras.layers.UpSampling2D(size=(16, 16))(pool_8)

    concat = tf.concat([x, up_1, up_2, up_4, up_8], axis=-1)
    output = Conv2D(filters, (1, 1), padding='same')(concat)
    return output

def gan_generator(input_shape=(48, 48, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(inputs)

    # 3 densely_residual_blocks
    for _ in range(3):
        x = densely_residual_block(x)

    # Enhancing block for multi-scale fusion
    x = enhancing_block(x)

    # Final convolution to generate enhanced output
    x = Conv2D(input_shape[2], (3, 3), padding='same', activation='tanh')(x)

    outputs = x + inputs
    return tf.keras.Model(inputs=inputs, outputs=outputs)
