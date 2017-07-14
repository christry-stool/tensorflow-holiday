import tensorflow as tf


def get_batch(tfrecords_file, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'text_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['text_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=1,
                                              capacity=500)
    image_batch = tf.cast(image_batch, dtype=tf.float32)
    label_batch = tf.one_hot(label_batch, depth=2)
    label_batch = tf.reshape(label_batch, [batch_size, 2])
    label_batch = tf.cast(label_batch, dtype=tf.float32)
    return image_batch, label_batch


def get_single_example(tfrecords_file):
    filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'text_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['text_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    label = tf.cast(img_features['label'], tf.int32)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                text_, label_ = sess.run([image, label])
                i += 1
                if i % 50 == 0:
                    print(i)

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

# get_single_example('tfrecords/val.tfrecords')

'''
text, label = get_batch('tfrecords/test.tfrecords', 64)
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:
            text_, label_ = sess.run([text, label])
            print(text_)
            print(label_)
            i += 1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''