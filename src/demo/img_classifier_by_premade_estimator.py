# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np

def input_fn():
    dataset_path = "/Users/liueq/Workspace/machine-learning/tensorflow-workspace/TensorFlow-Examples-master/src/demo/dataset/car"

    if not tf.gfile.IsDirectory(dataset_path):
        tf.logging.error("dataset path is not exist.")
        quit()
    walk = tf.gfile.Walk(dataset_path)
    labels = []
    x = []
    y = []
    for w in walk:
        if w[0] == dataset_path:
            labels = labels + w[1]
        if w[0].split('/')[-1] in labels:
            for pic in w[2]:
                full_pic = w[0] + '/' + pic
                file_reader = tf.read_file(full_pic, "file_reader")
                if pic.endswith('jpg') or pic.endswith('JPG'):
                    image_reader = tf.image.decode_jpeg(file_reader, channels=1, name="image_reader")
                elif pic.endswith('png') or pic.endswith('PNG'):
                # else:
                    image_reader = tf.image.decode_png(file_reader, channels=1, name="image_reader")

                # 处理图片
                float_img = tf.cast(image_reader, tf.float32)
                dims_expander = tf.expand_dims(float_img, 0)
                resized = tf.image.resize_bilinear(dims_expander, [28, 28])
                normalized = tf.divide(tf.subtract(resized, [0]), [255])

                x.append(normalized)
                y.append(
                    tf.convert_to_tensor(labels.index(w[0].split('/')[-1]), dtype=tf.int8, name='y'))

    train_x = x # tensors list
    train_y = y
    print("Train data ready")

    return {'x': tf.stack(train_x)}, tf.stack(train_y)

column = tf.feature_column.numeric_column("x", shape=(1, 28, 28, 1, ))

estimator = tf.estimator.DNNClassifier(
    feature_columns=[column],
    hidden_units=[4, 4],
    n_classes=2
)

estimator.train(input_fn=input_fn)
print("Training end.")