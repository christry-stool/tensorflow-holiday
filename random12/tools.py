import tensorflow as tf
import numpy as np


def loss(logits, labels, regularizer=None):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        if regularizer != None:
            regularizer_loss = tf.add_n(tf.get_collection('losses'), name='regularizer_loss')
            loss_value = tf.add(cross_entropy_mean, regularizer_loss, name='loss')
        else:
            loss_value = cross_entropy_mean
        tf.summary.scalar(scope+'/loss', loss_value)
        return loss_value


def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(scope+'/accuracy', accuracy_value)
        return accuracy_value


def optimizer(losses, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=losses, global_step=global_step)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        #     .minimize(loss=losses, global_step=global_step)
        return train_step


def moving_average(decay, global_step):
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        return variable_averages_op


def print_all_variables(sess, train_only=True):
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:
            t_vars = tf.global_variables()
        except:
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
        value = sess.run([v])
        print(value)

