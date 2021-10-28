import tensorflow as tf
import model as model
import os
import numpy as np
import data_augmentation as DA
import h5py
from vgg16 import Vgg16

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
MAX_EPOCH = 100

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'DoFP_InterpModel'

IMG_SIZE = (40, 40)
IMG_CHANNEL = 12


def fspecial_gauss(size, sigma, ch=3):
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    f = g / tf.reduce_sum(g)
    f_exp = tf.tile(f, [1, 1, ch, 1]) / 3.0
    return f_exp


def SSIM_LOSS(img1, img2, size=11, sigma=1.5, ch=3):
    window = fspecial_gauss(size, sigma, ch)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='SAME')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_mu2

    v1 = 2 * mu1_mu2 + C1
    v2 = mu1_sq + mu2_sq + C1

    value = (v1 * (2.0 * sigma12 + C2)) / (v2 * (sigma1_sq + sigma2_sq + C2))
    value = tf.reduce_mean(value)
    value = 1.0 - value
    return value


def VGG_Loss(RGB0, RGB0_GT):
    # with tf.Graph().as_default() as g:
        S1_VGG_in = RGB0
        S2_VGG_in = RGB0_GT
        vgg1 = Vgg16()
        with tf.name_scope("vgg1"):
            S1_FEAS = vgg1.build(S1_VGG_in)
        vgg2 = Vgg16()
        with tf.name_scope("vgg2"):
            S2_FEAS = vgg2.build(S2_VGG_in)
        loss_vgg = tf.reduce_mean(tf.square(S1_FEAS - S2_FEAS))
        return loss_vgg


def loss_fun(y_, y):
    y0_, y45_, y90_, y135_ = tf.split(y_, 4, -1)
    y0, y45, y90, y135 = tf.split(y, 4, -1)

    # MSE
    loss_mse = tf.reduce_mean(tf.square(y[:, :, :, 0:12] - y_))

    # SSIM
    loss_ssim_0 = SSIM_LOSS(y0, y0_)
    loss_ssim_45 = SSIM_LOSS(y45, y45_)
    loss_ssim_90 = SSIM_LOSS(y90, y90_)
    loss_ssim_135 = SSIM_LOSS(y135, y135_)
    loss_ssim = (loss_ssim_0 + loss_ssim_45 + loss_ssim_90 + loss_ssim_135) / 4.0

    # Stokes
    S0 = (y0 + y90 + y45 + y135) / 2.0
    S1 = y0 - y90
    S2 = y45 - y135

    S0_ = (y0_ + y90_ + y45_ + y135_) / 2.0
    S1_ = y0_ - y90_
    S2_ = y45_ - y135_

    loss_mse_s0 = tf.reduce_mean(tf.square(S0 - S0_))
    loss_mse_s1 = tf.reduce_mean(tf.square(S1 - S1_))
    loss_mse_s2 = tf.reduce_mean(tf.square(S2 - S2_))

    # VGG loss
    VGGLoss_0 = VGG_Loss(y0, y0_)
    VGGLoss_45 = VGG_Loss(y45, y45_)
    VGGLoss_90 = VGG_Loss(y90, y90_)
    VGGLoss_135 = VGG_Loss(y135, y135_)
    VGGLoss = (VGGLoss_0 + VGGLoss_45 + VGGLoss_90 + VGGLoss_135)/4.0

    loss = loss_mse + loss_mse_s0 + loss_mse_s1 + loss_mse_s2 + loss_ssim + VGGLoss
    return loss



def backward(train_data, train_labels, train_num):
    with tf.Graph().as_default() as g:
        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1])
            y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL])
        # forward
        y = model.forward(x, True)

        # learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   train_num // BATCH_SIZE,
                                                   LEARNING_RATE_DECAY, staircase=True)
        # loss function
        with tf.name_scope('loss'):
            loss = loss_fun(y_, y)
        # Optimizer
        # GradientDescent
        with tf.name_scope('train'):
            # Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # Save model
        saver = tf.train.Saver(max_to_keep=100)
        epoch = 0
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('-')[-2])

            while epoch < MAX_EPOCH:
                max_step = train_num // BATCH_SIZE
                listtmp = np.random.permutation(train_num)
                j = 0
                for i in range(max_step):
                    file = open("loss.txt", 'a')
                    ind = listtmp[j:j + BATCH_SIZE]
                    j = j + BATCH_SIZE
                    xs = train_data[ind, :, :, :]
                    ys = train_labels[ind, :, :, :]
                    mode = np.random.permutation(8)
                    xs = DA.data_augmentation(xs, mode[0])
                    ys = DA.data_augmentation(ys, mode[0])

                    _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                    file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (
                        epoch + 1, step, i + 1, max_step, loss_v))
                    file.close()
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_epoch_' + str(epoch + 1)),
                           global_step=global_step)
                epoch += 1


if __name__ == '__main__':
    data = h5py.File('./trainingData.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)
    input_npy = np.reshape(input_npy, (input_npy.shape[0], input_npy.shape[1], input_npy.shape[2], 1))
    train_num = input_npy.shape[0]
    output_data = data["outputs"]
    labels = np.transpose(output_data)
    backward(input_npy, labels, train_num)
