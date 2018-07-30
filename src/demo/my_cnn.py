# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import argparse

def loadImage(dir_name):
    """
    列出 dir 的所有子目录。
    返回 float 类型的图片数据; str 类型的 label 数据。两个列表一一对应。
    """
    print("loadImage")

    if not tf.gfile.IsDirectory(dir_name):
        print("{} is not directory".fromat(dir_name))
        return
    # sub_dir = tf.gfile.ListDirectory(dir_name)
    walk_list = tf.gfile.Walk(dir_name)
    categories = []
    dataset_x = []
    dataset_y = []
    for walk in walk_list:
        if walk[0] == dir_name:
            print ("do append category")
            categories = walk[1]
        # print (categories)

        sub_dir = walk[0].split('/')[-1]
        if sub_dir in categories:
            dataset_x = dataset_x + walk[2]
            for i in range(len(walk[2])):
                dataset_y.append(sub_dir)

    # print (dataset_x)
    # print (dataset_y)
    for (x, y) in zip(dataset_x, dataset_y):
        if debug: print x, y
        
        full_file_name = dir_name + '/' + y + '/' + x
        file_reader = tf.read_file(full_file_name, "file_reader")

        if x.endswith('jpg') or x.endswith('JPG'):
            image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
        elif x.endswith('png') or x.endswith('PNG'):
            image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='png_reader')
        
        float_caster = tf.cast(image_reader, tf.float32)

        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [299, 299]) # 最终每个图片得到的 tensor 矩阵大小
        normalized = tf.divide(tf.subtract(resized, [0]), [255]) # 分别是 input_mean, input_std

    print normalized
    tf.estimator.inputs.numpy_input_fn(
        x={'f1': normalized[]}
        y=dataset_y
    )
    return normalized, dataset_y

def input_fn():
    x, y = loadImage("/Users/liueq/Workspace/machine-learning/tensorflow-workspace/tutorial/src/images/image-retraining/city_photos")
    slice1 = tf.data.Dataset.from_tensor_slices(x)
    slice2 = tf.data.Dataset.from_tensor_slices(y)

    return slice1, slice2


def buildEstimator():
    print("printEstimator")

    # 定义 feature column
    column1= tf.feature_column.numeric_column("feature1", shape=(10, 10), dtype=tf.float32)
    column2 = tf.feature_column.numeric_column("feature2", shape=(10, 10), dtype=tf.float32)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[1024, 512, 256],
        feature_columns={
            column1,
            column2
        }
    )

    estimator.train(input_fn)

def train(estimator):
    print("Train")

def test():
    print("Test")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-D', '--debug', help='Debug mode')
    ap.add_argument('-d', '--directory', help='Training dataset folder')
    # ap.add_argument('-l', '--learning_rate', help='Learning rate')

    args = vars(ap.parse_args())
    path = args["directory"]
    debug = bool(args["debug"])
    if debug: 
        print ("%s" % path)

    input_x, input_y = loadImage(path) # python src/demo/my_cnn.py -d /Users/liueq/Workspace/machine-learning/tensorflow-workspace/tutorial/src/images/image-retraining/city_photos
    # buildEstimator()

