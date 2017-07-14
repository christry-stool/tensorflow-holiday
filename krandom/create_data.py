import tensorflow as tf
import numpy as np
import random
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

random_matrix1 = []
random_matrix2 = []
if os.path.exists('random_matrix1.txt'):
        print('exist')
        with open('random_matrix1.txt', 'r') as file:
            for line in file:
                line = line.strip('\n')
                random_matrix1.append(int(line))

        with open('random_matrix2.txt', 'r') as file:
            for line in file:
                line = line.strip('\n')
                random_matrix2.append(int(line))
else:
    for i in range(392):
        random_matrix1.append(random.randint(0, 255))
        random_matrix2.append(random.randint(0, 255))


def create_random_random1_example():
    plaintext = []
    ciphertext = []
    text = []
    for i in range(392):
        plaintext.append(random.randint(0, 255))
        ciphertext.append((plaintext[i] + random_matrix1[i]) % 256)
        text.append(plaintext[i])
        text.append(ciphertext[i])
    text = np.array(text, dtype=np.int8)
    return text, 0


def create_random_random2_example():
    plaintext = []
    ciphertext = []
    text = []
    for i in range(392):
        plaintext.append(random.randint(0, 255))
        ciphertext.append((plaintext[i] + random_matrix2[i]) % 256)
        text.append(plaintext[i])
        text.append(ciphertext[i])
    text = np.array(text, dtype=np.int8)
    return text, 1


def create_data(data_size):
    text_list = []
    label_list = []
    for i in range(data_size):
        if i % 2 == 0:
            text, label = create_random_random1_example()
        else:
            text, label = create_random_random2_example()
        text_list.append(text)
        label_list.append(label)
        if i % 100 == 0:
            print('created ', i+1, 'examples')
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

        writer.write(example.SerializeToString())
        if i % 200 == 0:
            print("writed ", i+1, 'examples')
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

filename = 'random_matrix1.txt'
output = open(filename, 'w')
for item in random_matrix1:
    item = str(item)
    output.write(item)
    output.write('\n')
output.close()

filename = 'random_matrix2.txt'
output = open(filename, 'w')
for item in random_matrix2:
    item = str(item)
    output.write(item)
    output.write('\n')
output.close()

'''

'''
filename = 'tfrecords/test2.tfrecords'
texts, labels = create_data(100000)
save_data(texts, labels, filename)
'''

filename = 'tfrecords/test2.tfrecords'
texts, labels = create_data(500000)
save_data(texts, labels, filename)



