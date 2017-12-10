"""
TF 提供了一种统一的TFRecord格式来存储数据。
通过tf.train.Example Protocol Buffer格式存储。
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/example.proto
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/feature.proto
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

result_file_name = "mnist.tfrecords"


def to_tf_record():
    mnist = input_data.read_data_sets(train_dir="../data/mnist")
    images = mnist.train.images
    labels = mnist.train.labels

    size = images.shape[1]
    num_examples = mnist.train.num_examples

    writer = tf.python_io.TFRecordWriter(result_file_name)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        # message Example { Features features = 1; };
        # message Features { map<string, Feature> feature = 1; };
        example = tf.train.Example(features=tf.train.Features(feature={
            # message Feature {
            #   oneof kind { BytesList bytes_list = 1; FloatList float_list = 2; Int64List int64_list = 3; }
            # };
            # message Int64List { repeated int64 value = 1 [packed = true]; }
            "size": tf.train.Feature(int64_list=tf.train.Int64List(value=[size, size])),
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[labels[index]])),
            # message BytesList { repeated bytes value = 1; }
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
        pass
    writer.close()
    pass


def read_tf_record():

    # 文件名队列
    file_name_queue = tf.train.string_input_producer([result_file_name])

    # 从队列中读取样例
    _, example = tf.TFRecordReader().read(file_name_queue)

    # 解析样例
    features = tf.parse_single_example(example, features={
        "size": tf.FixedLenFeature([2], tf.int64, default_value=[786, 786]),
        "label": tf.FixedLenFeature([], tf.float32),
        "image": tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features["image"], tf.uint8)
    label = tf.cast(features["label"], tf.int32)
    size = tf.cast(features["size"], tf.int32)

    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(100):
            size_now, label_now, image_now = sess.run([size, label, image])
            print("size: {} {},label: {}".format(size_now[0], size_now[1], label_now))
        coord.request_stop()
        coord.join(threads=threads)
        pass
    pass

if __name__ == '__main__':
    # to_tf_record()
    read_tf_record()
