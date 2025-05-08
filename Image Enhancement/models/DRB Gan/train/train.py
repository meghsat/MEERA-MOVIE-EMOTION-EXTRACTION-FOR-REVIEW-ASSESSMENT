import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.generator import build_generator
from models.discriminators import build_local_discriminator, build_global_discriminator
from train.losses import (
    wasserstein_d_loss, wasserstein_g_loss, vgg_feature_loss
)

IMG_SHAPE = (48, 48, 3)
BATCH_SIZE = 64
CRITIC_STEPS = 5           
GP_WEIGHT = 10.0          

gen   = build_generator(IMG_SHAPE)
d_loc = build_local_discriminator((32, 32, 3))
d_glb = build_global_discriminator(IMG_SHAPE)
opt_G = Adam(1e-4, beta_1=0.0, beta_2=0.9)
opt_D = Adam(1e-4, beta_1=0.0, beta_2=0.9)

@tf.function
def gradient_penalty(disc, real, fake):
    alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0.0, 1.0)
    inter = real + alpha * (fake - real)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(inter)
        pred = disc(inter)
    grads = gp_tape.gradient(pred, [inter])[0]
    gp = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    return tf.reduce_mean((gp - 1.0) ** 2)

@tf.function
def train_step(low_light, gt):
    for _ in range(CRITIC_STEPS):
        with tf.GradientTape(persistent=True) as tape_d:
            fake = gen(low_light, training=True)

            patch_real = tf.image.random_crop(gt,  size=[BATCH_SIZE, 32, 32, 3])
            patch_fake = tf.image.random_crop(fake, size=[BATCH_SIZE, 32, 32, 3])

            real_loc_logits = d_loc(patch_real, training=True)
            fake_loc_logits = d_loc(patch_fake, training=True)

            real_glb_logits = d_glb(gt,   training=True)
            fake_glb_logits = d_glb(fake, training=True)

            d_loss_loc = wasserstein_d_loss(real_loc_logits, fake_loc_logits)
            d_loss_glb = wasserstein_d_loss(real_glb_logits, fake_glb_logits)

            gp_loc = gradient_penalty(d_loc, patch_real, patch_fake)
            gp_glb = gradient_penalty(d_glb, gt, fake)

            d_total = d_loss_loc + d_loss_glb + GP_WEIGHT * (gp_loc + gp_glb)

        grads_d = tape_d.gradient(d_total, d_loc.trainable_variables + d_glb.trainable_variables)
        opt_D.apply_gradients(zip(grads_d, d_loc.trainable_variables + d_glb.trainable_variables))
        del tape_d

    with tf.GradientTape() as tape_g:
        fake = gen(low_light, training=True)

        fake_loc_logits = d_loc(tf.image.random_crop(fake, [BATCH_SIZE, 32, 32, 3]), training=False)
        fake_glb_logits = d_glb(fake, training=False)
        g_adv  = wasserstein_g_loss(fake_loc_logits) + wasserstein_g_loss(fake_glb_logits)

        g_feat = vgg_feature_loss(gt, fake)

        g_total = g_adv + 0.1 * g_feat     

    grads_g = tape_g.gradient(g_total, gen.trainable_variables)
    opt_G.apply_gradients(zip(grads_g, gen.trainable_variables))

    return {"d_loss": d_total, "g_loss": g_total}