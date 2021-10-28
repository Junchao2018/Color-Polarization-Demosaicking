import tensorflow as tf
import numpy as np

def Dense_block(x):
    for i in range(7):
        shape = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0 / shape[-1]))
        t = tf.layers.conv2d(x,16,3,(1,1),padding='SAME',kernel_initializer=w_init)
        t = tf.nn.relu(t)
        x = tf.concat([x,t],3)
    return x


def conv_layer(x, filter_num, name, is_training):
    with tf.name_scope('layer_%02d' %name):
        ch = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/(9.0*ch[-1])))
        output = tf.layers.conv2d(x, filter_num, 3, (1,1), padding='SAME',activation=None,kernel_initializer = w_init)
        output = tf.layers.batch_normalization(output, training=is_training)
        output = tf.nn.relu(output)
        return output



def PDCNN(x,pname,is_training):
    with tf.name_scope(name=pname):
        output_list = []
        output = x
        for i in range(6):
            output = conv_layer(output, 64, i, is_training)
            output_list.append(output)

        Mid_layer_output = []
        for j in range(1, 6):
            tmp = output_list[-1 - j]
            output = tf.concat([tmp, output], 3)
            output = conv_layer(output, 64, j + i, is_training)
            Mid_layer_output.append(output)

        output_layer = []
        with tf.name_scope('layer_final'):
            w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / (9.0 * 64)))
            for mid in Mid_layer_output:
                output = tf.layers.conv2d(mid, 12, 3, (1, 1), padding='SAME', use_bias=True, kernel_initializer=w_init)
                output_layer.append(output)

        output = tf.concat(output_layer, 3)

        Num = len(output_layer)
        Num = Num * 12
        with tf.name_scope('recons_layer'):
            w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / (9.0 * Num)))
            output = tf.layers.conv2d(output, 12, 1, (1, 1), padding='VALID', use_bias=False, kernel_initializer=w_init)

        return output

def forward(input_polar, input_mosaic, is_training):
    with tf.device("/gpu:0"):
        output_polar = PDCNN(input_polar, '12_channels', is_training)
        output_polar = tf.nn.relu(output_polar)
        output_mosaic = Dense_block(input_mosaic)
        output_mosaic = tf.layers.conv2d(output_mosaic, 12, 3, (1, 1), padding='SAME')
        output = output_mosaic + output_polar
        return output
