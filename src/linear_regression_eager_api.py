# -*- coding=utf8 -*-
"""
使用 eager execution 和不使用的基本逻辑是没有区别的。只是说 eager execution 展示了更多的细节。
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()


train_X = np.asarray([3.3, 4.4, 5.5, 6.17, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples = len(train_X)

learning_rate = 0.01
display_step = 100
num_steps = 1000

W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())

def linear_regression(inputs):
    return inputs * W + b

def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels , 2)) / (2 * n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

grad = tfe.implicit_gradients(mean_square_fn)

print("Initial cost= {:.9f}".format(
    mean_square_fn(linear_regression, train_X, train_Y)),
    "W=", W.numpy(), "b=", b.numpy())

for step in range(num_steps):
    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))

    if (step + 1) % display_step == 0 or step == 0:
        print ("Epoch:", "%04d" % (step + 1), "cost=",
               "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
               "W=", W.numpy(), "b=", b.numpy())

plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, np.array(W * train_X + b), label="Fitted line")
plt.legend()
plt.show()


