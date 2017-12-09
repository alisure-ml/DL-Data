"""
tf.train.batch()和tf.train.shuffle_batch()将单个的样例组织成batch的形式输出。
这两个函数都会生成一个队列，队列的入队操作是生成单个样例的方法，每次出队得到的是一个batch的样例。
"""
import tensorflow as tf
from my_3_string_input_producer import produce_data, read_example

# 产生模拟数据
produce_data()

# 获取文件列表
files = tf.train.match_filenames_once("data.tfrecords-*")

# 创建“文件列表输入队列”
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取数据
example, label = read_example(filename_queue)

batch_size = 3
# 队列的最大容量：太小会导致出队操作因为没有数据而被阻塞，太大会占用太多内存。
capacity = 1000 + 3 * batch_size

# 当队列长度等于容量时，将暂停入队操作，当小于容量时，将自动启动入队操作。
# example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
# example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size, capacity, min_after_dequeue=100)
# 通过设置num_threads可以指定多个线程同时执行入队操作。
# example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size, capacity, 100, num_threads=4)
# 如果需要多个线程处理不同的文件中的样例时，可以使用tf.train.shuffle_batch_join(),
# 此函数会从输入文件队列中获取不同的文件分配给不同的线程。
example_batch, label_batch = tf.train.shuffle_batch_join([(example, label)] * 5, batch_size, capacity, 100)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5):
        cur_batch_example, cur_batch_label = sess.run([example_batch, label_batch])
        print(cur_batch_example, cur_batch_label)

    coord.request_stop()
    coord.join(threads)
