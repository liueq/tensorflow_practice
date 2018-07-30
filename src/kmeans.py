# -*- coding:utf8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

# 定义 kmeans 参数
num_steps = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 784  # 28*28

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# 获取实例
kmeans = KMeans(inputs=X, num_clusters=k,
                distance_metric='cosine', use_mini_batch=True)

# 获得可修改的 training_graph 参数
(all_scores, cluster_idx, scores, cluster_centers_initialized,
 init_op, train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # 从 tuple 转换为 int
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

for i in range(1, num_steps + 1):
    # sess 运行了 3 个函数，分别是训练(train_op), 求 avg_distance, cluster_idx
    _, d, idx = sess.run(
        [train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print ("Step %i, Avg distance: %f" % (i, d))

counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]

labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
correct_prediction = tf.equal(
    cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
# reduce_mean, 求均值，第二个参数可以指定 axis，0代表纵向求均值，1代表横向求均值。如果制定了 axis，得到的是一个 vector
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(
    accuracy_op, feed_dict={X: test_x, Y: test_y}))
