
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

def wasserstein_d_loss(real_logits, fake_logits):

    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def wasserstein_g_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)



def vgg_feature_loss(y_true, y_pred, layer: str = "block3_conv3"):

    global _vgg
    if _vgg is None:
        base = VGG19(include_top=False, weights="imagenet")
        _vgg = tf.keras.Model(base.input, base.get_layer(layer).output)
        _vgg.trainable = False

    def _prep(img):
        img = (img + 1.0) * 127.5
        return preprocess_input(img)

    f_true = _vgg(_prep(y_true))
    f_pred = _vgg(_prep(y_pred))
    return tf.reduce_mean(tf.abs(f_true - f_pred))
