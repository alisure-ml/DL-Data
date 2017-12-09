"""
队列和变量类似，都是计算图上有状态的节点，修改队列状态的操作主要有Enqueue、EnqueueMany、Dequeue。

多个线程可以同时向一个队列中写元素，或者从一个队列中读元素。
"""
import tensorflow as tf


# 最基本的例子
def fifo_queue():
    # 创建一个队列，指定最多可以保存的元素数量和类型
    queue = tf.FIFOQueue(5, dtypes=[tf.int32])
    # 初始化队列中的元素
    # init = queue.enqueue_many(([1, 2, 3, 4],))
    init = queue.enqueue_many([[1, 2, 3, 4]])
    # 出队列
    x = queue.dequeue()
    # 操作
    y = x * x
    # 入队列
    power_enqueue = queue.enqueue([y])

    with tf.Session() as sess:
        init.run()
        for _ in range(10):
            v, __ = sess.run([x, power_enqueue])
            print(v)
        pass
    pass


# 同时取多个队列
def fifo_queue_multi_queue():
    # 创建一个队列，指定最多可以保存的元素数量和类型
    queue = tf.FIFOQueue(5, dtypes=[tf.int32, tf.float32, tf.int32])
    # 初始化队列中的元素
    init = queue.enqueue_many([[1, 2, 3, 4], [1.0, 2.0, 3.0, 4.9], [1, 2, 3, 4]])
    # 出队列
    x = queue.dequeue()
    # 操作
    x_power = x[0] * x[0]
    y = [x_power, tf.cast(x_power, tf.float32) - x[1], x_power + x[2]]
    # 入队列
    power_enqueue = queue.enqueue(y)

    with tf.Session() as sess:
        init.run()
        for _ in range(10):
            v, __ = sess.run([x, power_enqueue])
            print(v)
        pass
    pass


# 随机队列:从当前队列所有元素中随机选择一个
def random_shuffle_queue_multi_queue():
    # 创建一个队列，指定最多可以保存的元素数量和类型
    queue = tf.RandomShuffleQueue(5, min_after_dequeue=3, dtypes=[tf.int32, tf.float32, tf.int32])
    # 初始化队列中的元素
    init = queue.enqueue_many([[1, 2, 3, 4], [1.0, 2.0, 3.0, 4.9], [1, 2, 3, 4]])
    # 出队列
    x = queue.dequeue()
    # 操作
    x_power = x[0] * x[0]
    y = [x_power, tf.cast(x_power, tf.float32) - x[1], x_power + x[2]]
    # 入队列
    power_enqueue = queue.enqueue(y)

    with tf.Session() as sess:
        init.run()
        for _ in range(10):
            v, __ = sess.run([x, power_enqueue])
            print(v)
        pass
    pass


if __name__ == '__main__':
    random_shuffle_queue_multi_queue()
