import tensorflow as tf
import numpy as np
import random


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_random_k1_example():
    plaintext = []
    ciphertext = []
    text = []
    for i in range(392):
        plaintext.append(random.randint(0, 255))
        ciphertext.append((plaintext[i]+1) % 256)
        text.append(plaintext[i])
        text.append(ciphertext[i])
    text = np.array(text, dtype=np.int8)
    return text, 0


def create_random_k2_example():
    plaintext = []
    ciphertext = []
    text = []
    for i in range(392):
        plaintext.append(random.randint(0, 253))
        ciphertext.append((plaintext[i]+2) % 256)
        text.append(plaintext[i])
        text.append(ciphertext[i])
    text = np.array(text, dtype=np.int8)
    return text, 1


def create_data(data_size):
    text_list = []
    label_list = []
    for i in range(data_size):
        if i % 2 == 0:
            text, label = create_random_k1_example()
        else:
            text, label = create_random_k2_example()
        text_list.append(text)
        label_list.append(label)
        if i % 100 == 0:
            print('created ', i + 1, 'examples')
    examples = np.array([text_list, label_list])
    examples = examples.transpose()
    np.random.shuffle(examples)
    texts = np.array(examples[:, 0])
    labels = np.array(examples[:, 1], dtype=np.int32)
    return texts, labels


def save_data(texts, labels, filename):
    num_examples = len(labels)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num_examples):
        text_raw = texts[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(labels[i]),
            'text_raw': _bytes_feature(text_raw)
        }))
        if i % 200 == 0:
            print("writed ", i + 1, 'examples')
        writer.write(example.SerializeToString())
    writer.close()

'''
filename = 'tfrecords/test.tfrecords'
texts, labels = create_data(10000)
save_data(texts, labels, filename)
print('test examples created completely')
filename = 'tfrecords/val.tfrecords'
texts, labels = create_data(10000)
save_data(texts, labels, filename)
print('val examples created completely')
filename = 'tfrecords/train.tfrecords'
texts, labels = create_data(10000)
save_data(texts, labels, filename)
print('train examples created completely')
filename = 'tfrecords/train2.tfrecords'
texts, labels = create_data(100000)
save_data(texts, labels, filename)
print('train2 examples created completely')
'''

filename = 'tfrecords/test5.tfrecords'
texts, labels = create_data(500000)
save_data(texts, labels, filename)
print('train2 examples created completely')






