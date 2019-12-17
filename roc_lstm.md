# 双向LSTM实现字符识别



# RNN概述

Recurrent Neural Network - 循环神经网络，最早出现在20世纪80年代，主要是用于时序数据的预测和分类。它的基本思想是：前向将上一个时刻的输出和本时刻的输入同时作为网络输入，得到本时刻的输出，然后不断地重复这个过程。后向通过BPTT(Back Propagation Through Time)算法来训练得到网络的权重。RNN比CNN更加彻底的是，CNN通过卷积运算共享权重从而减少计算量，而RNN从头到尾所有的权重都是公用的，不同的只是输入和上一时刻的输出。RNN的缺点在于长时依赖容易被遗忘，从而使得长时依赖序列的预测效果较差。

LSTM(Long Short Memory)是RNN最著名的一次改进，它借鉴了人类神经记忆的长短时特性，通过门电路(遗忘门，更新门)的方式，保留了长时依赖中较为重要的信息，从而使得RNN的性能大幅度的提高。

为了提高LSTM的计算效率，学术界又提供了很多变体形式，最著名的要数GRU(Gated Recurrent Unit)，在减少一个门电路的前提下，仍然保持了和LSTM近似的性能，成为了语音和nlp领域的宠儿。

# 双向LSTM实现字符识别

下面的代码实现了一个双向的LSTM网络来进行mnist数据集的字符识别问题，双向的LSTM优于单向LSTM的是它可以同时利用过去时刻和未来时刻两个方向上的信息，从而使得最终的预测更加的准确。

Tensorflow提供了对LSTM Cell的封装，这里我们使用BasicLSTMCell，定义前向和后向的LSTM Cell:

```python?linenums
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
```

然后通过static_bidrectional_rnn函数将这两个cell以及时序输入x进行整合:

```python?linenums
outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
lstm_fw_cell,
lstm_bw_cell,
x,
dtype=tf.float32
)
```

完整的代码如下:

```python?linenums
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10
n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x, weights, biases):
x = tf.transpose(x, [1, 0, 2])
x = tf.reshape(x, [-1, n_input])
x = tf.split(x, n_steps)
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
lstm_fw_cell,
lstm_bw_cell,
x,
dtype=tf.float32
)
return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
step = 1
while step * batch_size < max_samples:
batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = batch_x.reshape((batch_size, n_steps, n_input))
sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
if step % display_step == 0:
acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
print ("Iter" + str(step * batch_size) + ", Minibatch Loss=" + \
"{:.6f}".format(loss) + ", Training Accuracy= " + \
"{:.5f}".format(acc))
step += 1
print ("Optimization Finishes!")

test_len = 50000
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print ("Testing accuracy:",
sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
```

这里选择了400000个sample进行训练，图像按行读入像素序列(总共n_step=28行)，每128个样本看成一个batch做一次BPTT，每10个batch打印一次training loss。

```python?linenums
Iter396800, Minibatch Loss=0.038339, Training Accuracy= 0.98438
Iter398080, Minibatch Loss=0.007602, Training Accuracy= 1.00000
Iter399360, Minibatch Loss=0.024104, Training Accuracy= 0.99219
Optimization Finishes!
```

取50000个样本作为测试集，准确率为：

```python?linenums
('Testing accuracy:', 0.98680007)
```

可以发现，双向LSTM做图像分类虽然也有不错的性能，但是还是比CNN略微逊色。主要原因应该还是因为图像数据属于层次性比较高的数据，CNN能够逐层抽取图像的层次特征，从而达到比较高的精度。但是可以想象，对于时序性比较强的无空间结构数据，RNN会有更加出色的表现。


本文来源：

> * [深度学习之循环神经网络RNN概述，双向LSTM实现字符识别](https://www.cnblogs.com/zdz8207/p/7468576.html)
