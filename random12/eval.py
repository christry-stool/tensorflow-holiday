import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import model
import tools
import input_data

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
BATCH_SIZE = 1
TRAINING_STEPS = 2
REGULARAZTION_RATE = 0.0001


def evaluate():
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

                    val_xs, val_ys = get_test_example()
                    yy, loss_value, acc_value = sess.run([predict_y, val_loss, val_acc],
                                                         feed_dict={x: val_xs, y_: val_ys})
                    global_score += acc_value
                    num_step += 1
                    if i % 1 == 0:
                        print("%d:After %s training step(s), validation accuracy = %g" % (i, global_step, acc_value))
                        #                        plot_images(val_xs, val_ys)

            except tf.errors.OutOfRangeError:
                print("global accuracy = %g" % (global_score / num_step))
                print('Done training -- epoch limit reached')

            finally:
                coord.request_stop()

            coord.join(threads)


def get_random_matrixs():
    filename1 = 'random_matrix1.txt'
    filename2 = 'random_matrix2.txt'

    random_matrix1 = []
    random_matrix2 = []

    with open(filename1, 'r') as file:
        for line in file:
            line = line.strip('\n')
            random_matrix1.append(int(line))

    with open(filename2, 'r') as file:
        for line in file:
            line = line.strip('\n')
            random_matrix2.append(int(line))

    print(random_matrix1)
    print(random_matrix2)
    return random_matrix1, random_matrix2


def get_test_example():
    random_matrix1, random_matrix2 = get_random_matrixs()
    images = []
    labels = []
    for i in range(25):
        text = []
        ctext = []
        image = []
        label = [0, 1] if i % 2 == 0 else [1, 0]
        for j in range(392):
            text.append(0)
            if i % 2 == 0:
                ctext.append((text[j] + random_matrix2[j]))
            else:
                ctext.append((text[j] + random_matrix1[j]))
            image.append(text[j])
            image.append(ctext[j])

        image = np.array(image)
        label = np.array(label)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    images = np.reshape(images, [-1, 28, 28])
    labels = np.reshape(labels, [-1, 2])
    return images, labels


def plot_images(images, labels):
    images = np.reshape(images, [-1, 28, 28])
    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        label = 0 if labels[i][0] == 1 else 1
        plt.title(label, fontsize=14)
        plt.subplots_adjust(top=1.5)
        image = np.reshape(images[i], [28, 28])
        plt.imshow(image, cmap=plt.cm.gray_r)
    plt.show()


images, labels = get_test_example()
plot_images(images, labels)
