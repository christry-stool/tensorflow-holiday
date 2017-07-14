import tensorflow as tf

import layers

# for cnn
def inference(input_tensor, n_classes, train=True, regularizer=None, evaluate=False):
    with tf.name_scope('cnn'):
        input_tensor = layers.conv('conv1', input_tensor, out_channels=32, kernel_size=[2, 2], strides=[1, 2, 2, 1], train=train)
        input_tensor = layers.pool('pool1', input_tensor, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], by_max=True)

        input_tensor = layers.conv('conv2', input_tensor, out_channels=64, kernel_size=[2, 2], strides=[1, 1, 1, 1], train=train)
        input_tensor = layers.batch_norm(input_tensor)
        input_tensor = layers.pool('pool2', input_tensor, kernel_size=[1, 2, 2, 1], strides=[1, 1, 1, 1], by_max=True)

        input_tensor = layers.full_connect('fc1', input_tensor, out_nodes=512, regularizer=regularizer)
        output_tensor = layers.full_connect_not_relu('fc2', input_tensor, out_nodes=n_classes, regularizer=regularizer)

        return output_tensor

# for dnn
def inference2(input_tensor, n_classes, train=True, regularizer=None, evaluate=False):
    input_tensor = layers.full_connect('fc1', input_tensor, out_nodes=512, regularizer=regularizer)
#    input_tensor = layers.full_connect('fc2', input_tensor, out_nodes=256, regularizer=regularizer)
#    input_tensor = layers.full_connect('fc3', input_tensor, out_nodes=128, regularizer=regularizer)
    output_tensor = layers.full_connect_not_relu('fc4', input_tensor, out_nodes=n_classes, regularizer=regularizer)
    return output_tensor
