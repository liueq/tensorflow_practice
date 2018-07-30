# -*- coding:utf8 -*-
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

# nn 结构参数
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# 使用 tf.data.Dataset api 读取数据
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels)).batch(batch_size)
dataset_iter = tfe.Iterator(dataset)

# 继承 tfe.Network，只需要定义 layer，以及组合 layer 即可
class NeuralNet(tfe.Network):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = self.track_layer(
            tf.layers.Dense(n_hidden_1, activation=tf.nn.relu)
        )
        self.layer2 = self.track_layer(
            tf.layers.Dense(n_hidden_2, activation=tf.nn.relu)
        )
        self.out_layer = self.track_layer(tf.layers.Dense(num_classes))
    
    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)

neural_net = NeuralNet()

# 计算 loss
def loss_fn(inference_fn, inputs, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels
    ))

# 计算 accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grad = tfe.implicit_gradients(loss_fn)

average_loss = 0.
average_acc = 0.
for step in range(num_steps):
    # 遍历 dataset
    try:
        d = dataset_iter.next()
    except StopIteration:
        dataset_iter = tfe.Iterator(dataset)
        d = dataset_iter.next()

    x_batch = d[0]
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # 调用函数计算 loss
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss

    # 调用函数计算 accuracy
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        print("Initial loss= {:.9f}".format(average_loss))
    
    # 调用函数进行 gradientdescent
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))

    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), "loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(neural_net, testX,testY)
print("Testest Accuracy: {:.4f}".format(test_acc))