# tensorflow实现LeNet-5模型


# LeNet-5简介

### 概述

#### 主要结构

LeNet5诞生于1994年，由Yann LeCun提出，充分考虑图像的相关性。当时结构的特点如下：

1. 每个卷积层包含三个部分：卷积（Conv）、池化（ave-pooling）、非线性激活函数（sigmoid）
2. MLP作为最终的分类器
3. 层与层之间稀疏连接减少计算复杂度

#### 其结构如下图所示：

![LeNet-5](./images/1576044235919.png)

### 各层介绍

LeNet-5共有7层，不包含输入，每层都包含可训练参数；每个层有多个Feature Map，每个FeatureMap通过一种卷积滤波器提取输入的一种特征，然后每个FeatureMap有多个神经元。

#### C1层是一个卷积层

输入图片：32 x 32
卷积核大小：5 x 5
卷积核种类：6
输出featuremap大小：28 x 28 （32-5+1）
神经元数量：28 x 28 x 6
可训练参数：（5 x 5+1）x 6（每个滤波器5x5=25个unit参数和一个bias参数，一共6个滤波器）
连接数：（5x5+1）x 6 x 28 x 28

#### S2层是一个下采样层

输入：28 x 28
采样区域：2 x 2
采样方式：4个输入相加，乘以一个可训练参数，再加上一个可训练偏置，结果通过sigmoid
采样种类：6
输出featureMap大小：14 x 14（28/2）
神经元数量：14 x 14 x 6
可训练参数：2 x 6（和的权+偏置）
连接数：（2 x 2 + 1）x 6 x 14 x 14

S2中每个特征图的大小是C1中特征图大小的1/4

#### C3层也是一个卷积层

输入：S2中所有6个或者几个特征map组合
卷积核大小：5 x 5
卷积核种类：16
输出featureMap大小：10 x 10
C3中的每个特征map是连接到S2中的所有6个或者几个特征map的，表示本层的特征map是上一层提取到的特征map的不同组合。
存在的一个方式是：C3的前6个特征图以S2中3个相邻的特征图子集为输入。接下来6个特征图以S2中4个相邻特征图子集为输入。然后的3个以不相邻的4个特征图子集为输入。最后一个将S2中所有特征图为输入。
可训练参数：6 x（3 x 25+1）+ 6 x（4 x 25 + 1）+ 3 x（4 x 25 + 1）+（25  x 6 + 1）= 1516
连接数：10 x 10 x 1516 = 151600

#### S4层是一个下采样层

输入：10x10
采样区域：2x2
采样方式：4个输入相加，乘以一个可训练参数，再加上一个可训练偏置，结果通过sigmoid
采样种类：16
输出featureMap大小：5x5（10/2）
神经元数量：5x5x16=400
可训练参数：2x16=32（和的权+偏置）
连接数：16x（2x2+1）x5x5=2000

S4中每个特征图的大小是C3中特征图大小的1/4

#### C5层是一个卷积层

输入：S4层的全部16个单元特征map（与s4全相连）
卷积核大小：5x5
卷积核种类：120
输出featureMap大小：1x1（5-5+1）

可训练参数/连接：120x（16x5x5+1）=48120

#### F6层全连接层

输入：c5 120维向量
计算方式：计算输入向量和权重向量之间的点积，再加上一个偏置，结果通过sigmoid函数
可训练参数:84x(120+1)=10164

# tensorflow代码实现

```python?linenums
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 训练数据
x = tf.placeholder("float", shape=[None, 784])

# 训练标签数据
y_ = tf.placeholder("float", shape=[None, 10])

# 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 第一层：卷积层
# 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))

conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))

# 移动步长为1, 使用全0填充
conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')

# 激活函数Relu去线性化
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

 
# 第二层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

 
# 第三层：卷积层
conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(
    stddev=0.1))  # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64

conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))

conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充

relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

 

# 第四层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

 

# 第五层：全连接层
fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))# 7*7*64=3136把前一层的输出变成特征向量

fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))

pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])

fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)



# 为了减少过拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)



# 第六层：全连接层
fc2_weights = tf.get_variable("fc2_weights", [1024, 10],                           initializer=tf.truncated_normal_initializer(stddev=0.1))  # 神经元节点数1024, 分类节点10

fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))

fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

 

# 第七层：输出层
# softmax
y_conv = tf.nn.softmax(fc2)

 

# 定义交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

 

# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

 

# tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值

# 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

 

# 用平均值来统计测试准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 

# 开始训练
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})  # 评估阶段不使用Dropout
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练阶段使用50%的Dropout

 

# 在测试数据上测试准确率
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

```

实验结果：在step 2000时精度达到0.99

___

 参考链接：

[python/Tensorflow实现LeNet](https://blog.csdn.net/Florentina_/article/details/79817497)

