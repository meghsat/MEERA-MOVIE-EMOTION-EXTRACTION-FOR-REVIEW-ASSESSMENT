

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def demographics_classifier(input_shape=(48, 48, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    age = Dense(128, activation='relu')(x)
    age_output = Dense(3, activation='softmax', name='age')(age) 
    gender = Dense(128, activation='relu')(x)
    gender_output = Dense(2, activation='softmax', name='gender')(gender)

    return tf.keras.Model(inputs=inputs, outputs=[age_output, gender_output])
