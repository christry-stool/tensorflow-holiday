import tensorflow as tf

import model
import tools
import input_data
import os
import matplotlib.pyplot as plt
import numpy as np

TFRECORDS_PATH = 'tfrecords/'
LOG_TRAIN_PATH = 'log/train'
LOG_VAL_PATH = 'log/val'
LOG2_TRAIN_PATH = 'log2/train'
LOG2_VAL_PATH = 'log2/val'
MODEL_SAVE_PATH = 'models/'
MODEL2_SAVE_PATH = 'models2/'
MODEL_NAME = 'model.ckpt'

WIDTH = 28
HEIGHT = 28
CHANNEL = 1
NUM_CLASSES = 2

LEARNING_RATE = 0.001
BATCH_SIZE = 100
TRAINING_STEPS = 100000
REGULARAZTION_RATE = 0.0001

# for cnn
def train():
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

    train_tfrecords = TFRECORDS_PATH + 'train2.tfrecords'
    val_tfrecords = TFRECORDS_PATH + 'val.tfrecords'

    train_image_batch, train_label_batch = input_data.get_batch(train_tfrecords, BATCH_SIZE)
    val_image_batch, val_label_batch = input_data.get_batch(val_tfrecords, BATCH_SIZE)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    if ckpt and ckpt.model_checkpoint_path:
        num_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        num_step = int(num_step)
        print(num_step)
        global_step = tf.Variable(num_step, trainable=False)
    else:
        global_step = tf.Variable(0, trainable=False)

#    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = model.inference(x, NUM_CLASSES, regularizer=None)
    train_loss = tools.loss(logits=y, labels=y_)
    train_acc = tools.accuracy(logits=y, labels=y_)
    train_op = tools.optimizer(train_loss, LEARNING_RATE, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        val_writer = tf.summary.FileWriter(LOG_VAL_PATH, sess.graph)

        try:
            for i in range(TRAINING_STEPS):
                if coord.should_stop():
                    break

                xs, ys = sess.run([train_image_batch, train_label_batch])
                _, loss_value, acc_value, step = sess.run([train_op, train_loss, train_acc, global_step],
                                                          feed_dict={x: xs, y_: ys})

                if i % 50 == 0:
                    print("After %d training step(s), loss on training batch is %g, accuracy is %g" %
                          (step, loss_value, acc_value))

                if i % 50 == 0:
                    summary_str = sess.run(summary_op, feed_dict={x: xs, y_: ys})
                    train_writer.add_summary(summary_str, step)

                if i % 200 == 0:
                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    val_loss_value, val_acc_value = sess.run([train_loss, train_acc], feed_dict={x: val_xs, y_: val_ys})
                    print("After %d training step(s), valuation loss is %g, accuracy is %g" %
                          (step, val_loss_value, val_acc_value))
                    summary_str = sess.run(summary_op, feed_dict={x: val_xs, y_: val_ys})
                    val_writer.add_summary(summary_str, step)

                if i % 1000 == 0 or step + 1 == TRAINING_STEPS:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()

        coord.join(threads)

# for dnn
def train2():
    ckpt = tf.train.get_checkpoint_state(MODEL2_SAVE_PATH)

    train_tfrecords = TFRECORDS_PATH + 'train2.tfrecords'
    val_tfrecords = TFRECORDS_PATH + 'val.tfrecords'

    train_image_batch, train_label_batch = input_data.get_batch(train_tfrecords, BATCH_SIZE)
    val_image_batch, val_label_batch = input_data.get_batch(val_tfrecords, BATCH_SIZE)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    if ckpt and ckpt.model_checkpoint_path:
        num_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        num_step = int(num_step)
        print(num_step)
        global_step = tf.Variable(num_step, trainable=False)
    else:
        global_step = tf.Variable(0, trainable=False)
#    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = model.inference2(x, NUM_CLASSES, regularizer=None)
    train_loss = tools.loss(logits=y, labels=y_)
    train_acc = tools.accuracy(logits=y, labels=y_)
    train_op = tools.optimizer(train_loss, LEARNING_RATE, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG2_TRAIN_PATH, sess.graph)
        val_writer = tf.summary.FileWriter(LOG2_VAL_PATH, sess.graph)

        try:
            for i in range(TRAINING_STEPS):
                if coord.should_stop():
                    break

                xs, ys = sess.run([train_image_batch, train_label_batch])
                _, loss_value, acc_value, step = sess.run([train_op, train_loss, train_acc, global_step],
                                                          feed_dict={x: xs, y_: ys})

                if i % 50 == 0:
                    print("After %d training step(s), loss on training batch is %g, accuracy is %g" %
                          (step, loss_value, acc_value))

                if i % 50 == 0:
                    summary_str = sess.run(summary_op, feed_dict={x: xs, y_: ys})
                    train_writer.add_summary(summary_str, step)

                if i % 200 == 0:
                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    val_loss_value, val_acc_value = sess.run([train_loss, train_acc], feed_dict={x: val_xs, y_: val_ys})
                    print("After %d training step(s), valuation loss is %g, accuracy is %g" %
                          (step, val_loss_value, val_acc_value))
                    summary_str = sess.run(summary_op, feed_dict={x: val_xs, y_: val_ys})
                    val_writer.add_summary(summary_str, step)

                if i % 1000 == 0 or step + 1 == TRAINING_STEPS:
                    saver.save(sess, os.path.join(MODEL2_SAVE_PATH, MODEL_NAME), global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()

        coord.join(threads)

# test function for cnn
def validate():
    val_tfrecords = TFRECORDS_PATH + 'test5.tfrecords'

    val_image_batch, val_label_batch = input_data.get_batch(val_tfrecords, BATCH_SIZE, num_epochs=1)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    y = model.inference(x, NUM_CLASSES, evaluate=False)
    predict_y = tf.argmax(y, 1)
    val_loss = tools.loss(logits=y, labels=y_)
    val_acc = tools.accuracy(logits=y, labels=y_)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            global_score = 0.
            num_step = 0
            try:
                for i in range(TRAINING_STEPS):
                    if coord.should_stop():
                        break

                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    yy, loss_value, acc_value = sess.run([predict_y, val_loss, val_acc],
                                                         feed_dict={x: val_xs, y_: val_ys})
                    global_score += acc_value
                    num_step += 1
                    if i % 5 == 0:
                        print("in the %dth batch:After %s training step(s), test accuracy = %g" % (i, int(global_step) - 1, acc_value))
#                        plot_images(val_xs, val_ys)

            except tf.errors.OutOfRangeError:
                print("global accuracy = %g" % (global_score / num_step))
                print('Done testing -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)

# test function for dnn
def validate2():
    val_tfrecords = TFRECORDS_PATH + 'test3.tfrecords'

    val_image_batch, val_label_batch = input_data.get_batch(val_tfrecords, BATCH_SIZE, num_epochs=1)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, WIDTH, HEIGHT, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='y-input')

    y = model.inference2(x, NUM_CLASSES, evaluate=False)
    predict_y = tf.argmax(y, 1)
    val_loss = tools.loss(logits=y, labels=y_)
    val_acc = tools.accuracy(logits=y, labels=y_)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(MODEL2_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            global_score = 0.
            num_step = 0
            try:
                for i in range(TRAINING_STEPS):
                    if coord.should_stop():
                        break

                    val_xs, val_ys = sess.run([val_image_batch, val_label_batch])
                    yy, loss_value, acc_value = sess.run([predict_y, val_loss, val_acc],
                                                         feed_dict={x: val_xs, y_: val_ys})
                    global_score += acc_value
                    num_step += 1
                    if i % 5 == 0:
                        print("in the %dth batch:After %s training step(s), test accuracy = %g" % (i, int(global_step)-1, acc_value))
#                        plot_images(val_xs, val_ys)

            except tf.errors.OutOfRangeError:
                print("global accuracy = %g" % (global_score / num_step))
                print('Done testing -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)


def plot_images(images, labels):

    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        label = 0 if labels[i][0] == 1 else 1
        plt.title(label, fontsize=14)
        plt.subplots_adjust(top=1.5)
        image = np.reshape(images[i], [28, 28])
        plt.imshow(image)
    plt.show()

# train2()
# validate2()
validate()
# train()
