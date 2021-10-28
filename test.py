import tensorflow as tf
import model as model
import numpy as np
import math
import h5py
import scipy.io


MODEL_SAVE_PATH = './model/'
IMG_CHANNEL = 1
EPOCH = 100
IMG_SIZE = (2048,1848)


def test(test_polar, test_mosaic, N):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [1, IMG_SIZE[0], IMG_SIZE[1], 12])
        xm = tf.placeholder(tf.float32, [1, IMG_SIZE[0], IMG_SIZE[1], 1])
        y = model.forward(x, xm, False)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt:
                ckpt.model_checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                ImgOut = np.zeros([N, IMG_SIZE[0], IMG_SIZE[1], 12], dtype=np.float32)
                for i in range(N):
                    print(i)
                    output = sess.run(y, feed_dict={x: test_polar[i:i+1, :, :, :], xm: test_mosaic[i:i+1, :, :,:]})
                    ImgOut[i, :, :, :] = np.array(output)
                scipy.io.savemat('Results.mat', {'ImgOut': ImgOut})
            else:
                print("No checkpoint is found.")
                return

if __name__=='__main__':
    data = h5py.File('./validationData_polar.mat')
    input_polar = data["inputs"]
    input_polar = np.transpose(input_polar)

    train_num = input_polar.shape[0]

    data = h5py.File('./validationData_mosaic.mat')
    input_mosaic = data["inputs"]
    input_mosaic = np.transpose(input_mosaic)
    input_mosaic = np.reshape(input_mosaic,
                              (input_mosaic.shape[0], input_mosaic.shape[1], input_mosaic.shape[2], 1))

    test(input_polar, input_mosaic, train_num)

