# MNIST数据集分类


# 示例代码

本文主要介绍使用TensorFlow实现一个传统多层神经网络用于MNIST数据集分类。
主要代码如下：

```python?linenums
""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
# 导入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 导入tf
import tensorflow as tf

# Parameters
# 设定各种超参数
learning_rate = 0.1 # 学习率
num_steps = 500   # 训练500次
batch_size = 128  # 每批次取128个样本训练
display_step = 100  # 每训练100步显示一次

# Network Parameters
# 设定网络的超参数
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
# tf图的输入，因为不知道到底输入大小是多少，因此设定占位符
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
# 初始化w和b
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
# 创建模型
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    # 隐藏层1，全连接了256个神经元
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    # 隐藏层2，全连接了256个神经元
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    # 最后作为输出的全连接层，对每一分类连接一个神经元
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
# 开启模型
# 输入数据X，得到得分向量logits
logits = neural_net(X)
# 用softmax分类器将得分向量转变成概率向量
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
# 定义损失和优化器
# 交叉熵损失, 求均值得到---->loss_op
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# 优化器使用的是Adam算法优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 最小化损失得到---->可以训练的train_op
train_op = optimizer.minimize(loss_op)

# Evaluate model
# 评估模型
# tf.equal() 逐个元素进行判断，如果相等就是True，不相等，就是False。
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# tf.cast() 数据类型转换----> tf.reduce_mean() 再求均值
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
# 初始化这些变量（作用比如说，给他们分配随机默认值）
init = tf.global_variables_initializer()

# Start training
# 现在开始训练啦！
with tf.Session() as sess:

    # Run the initializer
    # 运行初始化器
    sess.run(init)

    for step in range(1, num_steps+1):
        # 每批次128个训练，取出这128个对应的data：x；标签：y
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        # train_op是优化器得到的可以训练的op，通过反向传播优化模型
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        # 每100步打印一次训练的成果
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # 计算每批次的是损失和准确度
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    # 看看在测试集上，我们的模型表现如何
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
```

本文来源

> * [TensorFlow入门示例教程](https://www.cnblogs.com/kongweisi/p/10996383.html)

