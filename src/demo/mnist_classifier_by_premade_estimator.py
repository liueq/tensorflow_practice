# -*- coding:utf8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

column = tf.feature_column.numeric_column("images", shape=(1, 784, ), dtype=tf.int32)

estimator = tf.estimator.DNNClassifier(
    feature_columns=[column],
    hidden_units=[256, 256],
    n_classes=10,
    model_dir="src/demo/checkpoint/mnist"
)

print mnist.train.images.shape
print mnist.train.labels.shape

train_y = np.asarray(mnist.train.labels, dtype=np.int32)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, 
    y=train_y,
    batch_size=128,
    num_epochs=None,
    shuffle=True
)

estimator.train(input_fn)