# -*- coding:utf8 -*-
"""
使用 Pre-made estimator:  DNNClassifer 对 iris 数据进行训练。

输入： 
由于本身就是 numeric，所以使用 numpy_array 的形式从文件中读入；
estimator 的输入是 feature_columns，所以这里使用了 numeric_column，需要定义 shape 和 numpy_array 一致。

model：
Estimator 主要定义3个参数，feature_columns, hidden_units, n_classes
需要定义 feature_columns，使用之前定义的 numeric_column。hidden_units 定义了隐藏层数，这里可以随意。

input_fn:
Estimator 的输入函数，数据真正输入的地方。x 需要用 dict 的形式，和之前定义的 feature_column 一一对应。
y 直接用 numpy_array 即可。
shuffle 必须有，如果是训练，设置为 True；eval 设置为 False。
num_epochs 必须有，如果是 None，默认用 1000。如果不设置，默认是 1，训练效果极差。
batch_size 随意，一般是 10.

eval 输出：
以 dict 的形式，有以下 keys
"average_loss"
"accuracy"
"global_step"
"loss"

保存训练好的模型，只需要在 estimator 中指定 model_dir 即可。
保存的格式是 checkpoint，跨平台通用。但是和 SavedModel 有区别。

"""
import tensorflow as tf
import numpy as np

iris_training = "src/demo/dataset/iris_training.csv"
iris_test = "src/demo/dataset/iris_test.csv"
training = np.loadtxt(fname=iris_training, dtype=np.float32, delimiter=',', skiprows=1)
test = np.loadtxt(fname=iris_training, dtype=np.float32, delimiter=',', skiprows=1)

train_x = training[:, 0:-1]
train_y = training[:, -1]
train_y = train_y.astype(np.int32)

test_x = test[:, 0:-1]
test_y = test[:, -1].astype(np.int32)

column = tf.feature_column.numeric_column('x', shape=(4, ))

estimator = tf.estimator.DNNClassifier(
    feature_columns=[column],
    hidden_units=[256, 256],
    n_classes=3,
    model_dir="src/demo/checkpoint/iris"
)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': train_x},
    y = train_y,
    shuffle=True,
    num_epochs=None, # 如果不设置 num_epochs，那么 global_step = 1，训练效果非常差
    batch_size=10
)

# train_dict = estimator.train(input_fn, steps=1000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': test_x},
    y = test_y,
    shuffle=False
)

eval_dict = estimator.evaluate(test_input_fn)

for k, v in eval_dict.iteritems():
    print k, v
