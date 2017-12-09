"""
使用tf.train.match_filenames_once()来获取符合一个正则表达式的所有文件。
得到的文件列表名可以通过tf.train.string_input_producer()进行有效的管理。
tf.train.string_input_producer()：
    1.使用“初始化时提供的文件列表”创建一个队列。
    2.通过设置shuffle参数，支持随机打乱文件列表中文件出队的顺序。
    3.当一个输入队列中的文件都被处理完后，它会将初始化时提供的文件列表中的文件全部重新加入队列。
    4.可以设置num_epochs参数限制加载初始文件列表的最大轮数。当所有文件都已经使用了设置的最大轮数后，继续读取文件会报错。
"""
import tensorflow as tf


def produce_data():
    num_shards = 2
    instances_per_shard = 2
    for i in range(num_shards):
        file_name = "data.tfrecords-{}-of-{}".format(i, num_shards)
        writer = tf.python_io.TFRecordWriter(file_name)
        for j in range(instances_per_shard):
            example = tf.train.Example(features=tf.train.Features(feature={
                "i": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                "j": tf.train.Feature(int64_list=tf.train.Int64List(value=[j]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
    pass


def read_example(filename_queue):
    _, serialized_example = tf.TFRecordReader().read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={"i": tf.FixedLenFeature([], tf.int64),
                                                                     "j": tf.FixedLenFeature([], tf.int64)})
    return features["i"], features["j"]


def string_input_producer():
    # 获取文件列表
    files = tf.train.match_filenames_once("data.tfrecords-*")

    # 创建“文件列表输入队列”
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    # 读取样例
    features_i, features_j = read_example(filename_queue)

    with tf.Session() as sess:
        """
        GLOBAL_VARIABLES
            Key to collect Variable objects that are global (shared across machines). 
            Default collection for all variables, except local ones.

        LOCAL_VARIABLES
            Key to collect local variables that are local to the machine and are not saved/restored.
        """
        # 初始化match_filenames_once所需的局部变量
        sess.run(tf.local_variables_initializer())
        print(sess.run(files))

        # 协同多线程
        coord = tf.train.Coordinator()
        # 启动所有线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(6):
            print(sess.run([features_i, features_j]))

        # 请求线程退出并等待所有线程退出
        coord.request_stop()
        coord.join(threads)
    pass

if __name__ == '__main__':
    produce_data()
    string_input_producer()
