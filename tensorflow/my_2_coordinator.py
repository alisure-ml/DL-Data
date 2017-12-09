"""
提供了tf.train.Coordinator和tf.train.QueueRunner两个类来完成“多线程协同”的功能
"""
import tensorflow as tf
import threading
import time


"""
tf.train.Coordinator主要用于协同多个线程一起停止，并提供了should_stop,request_stop,join三个函数
1.在启动线程之前，需要创建一个tf.train.Coordinator类，并将这个类传入到每一个创建的线程中。
2.每一个启动的线程需要一直查询should_stop函数，当返回True时，当前线程需要退出。
3.每一个启动的线程可以通过request_stop函数来通知其他线程退出。
    （当一个线程调用了request_stop后，其他线程调用should_stop会返回True）
"""


def coordinator():

    # 线程内执行的函数
    def my_loop(coord, count, t_id):
        # 判断当前线程是否需要退出
        while not coord.should_stop():
            for index in range(count):
                print("{} {}".format(t_id, index))
                time.sleep(1)
                pass
            # 通知其他线程退出
            coord.request_stop()
            print("stop {}".format(t_id))
        pass

    # 声明Coordinator类协同多个线程
    coord = tf.train.Coordinator()

    # 创建线程
    threads = [threading.Thread(target=my_loop, args=(coord, i + 1, i)) for i in range(3)]

    # 启动所有线程
    for t in threads:
        t.start()

    # 等待所有线程退出
    coord.join(threads)
    pass


"""
tf.train.QueueRunner用于启动多个线程操作同一个队列
"""


def queue_runner():
    # 声明一个FIFO队列
    queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32])

    # 入队操作
    enqueue_op = queue.enqueue(tf.random_uniform(shape=[1], dtype=tf.float32)[0])

    # 通过QueueRunner创建多个线程运行队列的入队操作
    queue_runner_tf = tf.train.QueueRunner(queue=queue, enqueue_ops=[enqueue_op, enqueue_op, enqueue_op])

    # 将QueueRunner加入到计算图中的指定集合，默认collection为QUEUE_RUNNERS
    tf.train.add_queue_runner(queue_runner_tf, collection="a")

    with tf.Session() as sess:
        # 协同多个线程
        coord = tf.train.Coordinator()

        # 启动所有线程，否则没有线程运行入队操作，当调用出队操作时，程序会一直等待。
        # 启动指定集合中的QueueRunner，默认collection为QUEUE_RUNNERS
        threads = tf.train.start_queue_runners(sess=sess, coord=coord, collection="a")

        for _ in range(4):
            print(sess.run(queue.dequeue()))

        # 请求停止所有线程并等待退出
        coord.request_stop()
        coord.join(threads=threads)
    pass

if __name__ == '__main__':
    queue_runner()
