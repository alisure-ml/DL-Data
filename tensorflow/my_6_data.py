import numpy as np
import tensorflow as tf


def demo():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset1.output_types)  # ==> "tf.float32"
    print(dataset1.output_shapes)  # ==> "(10,)"

    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]),
                                                   tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
    print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
    print(dataset2.output_shapes)  # ==> "((), (100,))"

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"


    dataset4 = tf.data.Dataset.from_tensor_slices({"a": tf.random_uniform([4]),
                                                   "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
    print(dataset4.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset4.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

    pass


sess = tf.Session()


def one_shot():
    dataset_one_hot = tf.data.Dataset.range(100)
    iterator = dataset_one_hot.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
        value = sess.run(next_element)
        assert i == value

    pass


def initializable():

    # 数据集
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)

    # 迭代器：可初始化的
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # 初始化迭代器
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)

    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        pass

    pass


def reinitializable():
    """
    A reinitializable iterator can be initialized from multiple different Dataset objects.
    For example, you might have a training input pipeline that uses random perturbations to the input images
     to improve generalization, and a validation input pipeline that evaluates predictions on unmodified data.
     These pipelines will typically use different Dataset objects that have the same structure
     (i.e. the same types and compatible shapes for each component).
    :return:
    """

    # 训练集和测试集
    training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -1, 1, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # 迭代器：每次送入数据具体结构的迭代器
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # 训练集初始化器
    training_init_op = iterator.make_initializer(training_dataset)

    # 测试集初始化器
    validation_init_op = iterator.make_initializer(validation_dataset)

    for _ in range(20):
        # 初始化训练集数据，从而使用训练集数据
        sess.run(training_init_op)
        for _ in range(100):
            _r = sess.run(next_element)

        # 初始化测试集数据，从而使用测试集数据
        sess.run(validation_init_op)
        for _ in range(50):
            _r = sess.run(next_element)

        pass

    pass


def feedable():
    # 训练集和测试集
    training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -1, 1, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # 迭代器：handle决定使用那个Dataset
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types,
                                                   training_dataset.output_shapes)
    next_element = iterator.get_next()

    # 初始化数据集的迭代器
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # 获取代表迭代器的string
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    while True:
        # 使用特定的Dataset
        for _ in range(200):
            _r = sess.run(next_element, feed_dict={handle: training_handle})
            pass

        # 初始化验证集的迭代器是为了运行完整的一代：Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            _r = sess.run(next_element, feed_dict={handle: validation_handle})
            pass

        pass

    pass


def get_next_and_out_of_range_error():

    def model(x):
        return tf.add(x, x)

    # 数据集
    dataset = tf.data.Dataset.range(5)

    # 初始化的迭代器
    iterator = dataset.make_initializable_iterator()

    # The Iterator.get_next() method returns one or more tf.Tensor objects
    # that correspond to the symbolic next element of an iterator.
    next_element = iterator.get_next()

    # 模型
    result = model(next_element)

    sess.run(iterator.initializer)

    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"

    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        print("End of dataset")

    # 重新初始化
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(result))
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
        pass

    # 数据集
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))

    # zip数据集
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

    # 可初始化的迭代器
    iterator = dataset3.make_initializable_iterator()

    # 初始化
    sess.run(iterator.initializer)

    # ???
    next1, (next2, next3) = iterator.get_next()

    pass


def saving_iterator_state(path_to_checkpoint=""):
    # 数据集
    dataset = tf.data.Dataset.range(5)

    # 初始化的迭代器
    iterator = dataset.make_initializable_iterator()

    saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

    # saver
    saver = tf.train.Saver()

    # 保存模型
    saver.save(sess, save_path=path_to_checkpoint)

    # 恢复模型
    saver.restore(sess, path_to_checkpoint)

    pass


def read_from_np_small_dataset():
    """
    需要把数据加入到图中作为tf.constant()的内容，所以只能用于小数据
    If all of your input data fit in memory, the simplest way to create a Dataset from
    them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().
    :return:
    """

    # 读取数据
    with np.load("/var/data/training_data.npy") as data:
        features = data["features"]
        labels = data["labels"]
        assert features.shape[0] == labels.shape[0]

    # Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    pass


def read_from_np_large_dataset():
    """
    可以用于大数据集
    :return:
    """

    # 读取数据
    with np.load("/var/data/training_data.npy") as data:
        features = data["features"]
        labels = data["labels"]
        assert features.shape[0] == labels.shape[0]

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

    # 迭代器
    iterator = dataset.make_initializable_iterator()

    # 初始化迭代器
    sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})

    pass


def read_from_tfrecord():
    # 从单一的、预知的源读取数据
    # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    # dataset = tf.data.TFRecordDataset(filenames)

    # 根据filenames切换读取数据的源
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    # 从训练数据集初始化数据
    training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

    # 从验证数据集初始化数据
    validation_filenames = ["/var/data/validation1.tfrecord", "/var/data/validation2.tfrecord"]
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
    pass


def read_from_text():
    # 1
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
    dataset = tf.data.TextLineDataset(filenames)

    # 2
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = dataset.flat_map(lambda filename: (tf.data.TextLineDataset(filename).skip(1)
                                                 .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

    pass


def read_from_csv():
    filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]

    # 8列：float
    record_defaults = [tf.float32] * 8  # Eight required float columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)

    # 8列：float
    record_defaults = [[0.0]] * 8
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)

    # 指定n列
    record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2, 4])

    pass


def map_parse_features():

    # 解析样本：提取指定数据
    def _parse_function(example_proto):
        features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["image"], parsed_features["label"]

    # Creates a dataset that reads all of the examples from two files, and extracts the image and label features.
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    # map
    dataset = dataset.map(_parse_function)

    pass


def map_parse_image():

    # 解析样本：读取数据并进行处理
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # 数据
    filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg"])
    labels = tf.constant([0, 37])

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

    pass


def map_parse_cv2():
    import cv2

    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
        image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
        return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg"]
    labels = [0, 37]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label: tuple(tf.py_func(_read_py_function,
                                                                   [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    pass


def batch():
    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))

    # batch: works for tensors that all have the same size.
    batched_dataset = dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
    pass


def batch_padding():
    """
    ???
    :return:
    """
    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=[None])

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0], [5, 5, 5, 5, 5, 0, 0], [6, 6, 6, 6, 6, 6, 0], [7, 7, 7, 7, 7, 7, 7]]

    pass


def training_workflow_1():
    # 1.数据重复10次，每批32个样本
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(...)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    pass


def training_workflow_2():
    # 2.数据重复10次，每批32个样本
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(...)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # 计算10代
    for _ in range(10):
        sess.run(iterator.initializer)
        while True:
            try:
                sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break

        pass

    pass


def training_workflow_3_shuffle():
    #
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(...)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    pass


def training_workflow_4():

    def model_function(_next_example, _next_label):
        _loss = ""
        return _loss

    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.map(...)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(10)
    iterator = dataset.make_one_shot_iterator()

    next_example, next_label = iterator.get_next()
    loss = model_function(next_example, next_label)

    training_op = tf.train.AdagradOptimizer(...).minimize(loss)

    with tf.train.MonitoredTrainingSession(...) as sess:
        while not sess.should_stop():
            sess.run(training_op)
        pass

    pass


def training_workflow_5():

    def dataset_input_fn():
        filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
        dataset = tf.data.TFRecordDataset(filenames)

        # Use `tf.parse_single_example()` to extract data from a `tf.Example`
        # protocol buffer, and perform any additional per-record preprocessing.
        def parser(record):
            keys_to_features = {
                "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
                "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
            parsed = tf.parse_single_example(record, keys_to_features)

            # Perform additional preprocessing on the parsed data.
            image = tf.image.decode_jpeg(parsed["image_data"])
            image = tf.reshape(image, [299, 299, 1])
            label = tf.cast(parsed["label"], tf.int32)

            return {"image_data": image, "date_time": parsed["date_time"]}, label

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label tensor for each example.
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(32)
        dataset = dataset.repeat(10)

        # Each element of `dataset` is tuple containing a dictionary of features
        # (in which each value is a batch of values for that feature), and a batch of labels.
        return dataset

    dataset_input_fn()

    pass

if __name__ == '__main__':

    saving_iterator_state()

    pass
