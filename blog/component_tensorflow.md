# TensorFlow组件介绍


# 引言

TensorFlow由谷歌人工智能团队谷歌大脑（Google Brain）开发和维护，拥有包括TensorFlow Hub、TensorFlow Lite、TensorFlow Research Cloud在内的多个项目以及各类应用程序接口（Application Programming Interface, API）  。自2015年11月9日起，TensorFlow依据阿帕奇授权协议（Apache 2.0 open source license）开放源代码。TensorFlow是一个基于数据流编程（dataflow programming）的符号数学系统，被广泛应用于各类机器学习（machine learning）算法的编程实现，其前身是谷歌的神经网络算法库DistBelief。Tensorflow拥有多层级结构，可部署于各类服务器、PC终端和网页并支持GPU和TPU高性能数值计算，被广泛应用于谷歌内部的产品开发和各领域的科学研究。

# 组件与工作原理

## 核心组件

分布式TensorFlow的核心组件（core runtime）包括：分发中心（distributed master）、执行器（dataflow executor/worker service）、内核应用（kernel implementation）和最底端的设备层（device layer）/网络层（networking layer）。

分发中心从输入的数据流图中剪取子图（subgraph），将其划分为操作片段并启动执行器。分发中心处理数据流图时会进行预设定的操作优化，包括公共子表达式消去（common subexpression elimination）、常量折叠（constant folding）等。

执行器负责图操作（graph operation）在进程和设备中的运行、收发其它执行器的结果。分布式TensorFlow拥有参数器（parameter server）以汇总和更新其它执行器返回的模型参数。执行器在调度本地设备时会选择进行并行计算和GPU加速。

内核应用负责单一的图操作，包括数学计算、数组操作（array manipulation）、控制流（control flow）和状态管理操作（state management operations）。内核应用使用Eigen执行张量的并行计算、cuDNN库等执行GPU加速、gemmlowp执行低数值精度计算，此外用户可以在内核应用中注册注册额外的内核（fused kernels）以提升基础操作，例如激励函数和其梯度计算的运行效率。

单进程版本的TensorFlow没有分发中心和执行器，而是使用特殊的会话应用（Session implementation）联系本地设备。TensorFlow的C语言API是核心组件和用户代码的分界，其它组件/API均通过C语言API与核心组件进行交互。

![tf_stru](./images/tf_stru.png)

## 低阶API

### 张量（tf.Tensor）

张量是TensorFlow的核心数据单位，在本质上是一个任意维的数组。可用的张量类型包括常数、变量、张量占位符和稀疏张量  。这里提供一个对各类张量进行定义的例子：

```python?linenums
import numpy as np
import tensorflow as tf
# tf.constant(value, dtype=None, name='Const', verify_shape=False)
tf.constant([0, 1, 2], dtype=tf.float32) # 定义常数
# tf.placeholder(dtype, shape=None, name=None)
tf.placeholder(shape=(None, 2), dtype=tf.float32) # 定义张量占位符
#tf.Variable(<initial-value>, name=<optional-name>)
tf.Variable(np.random.rand(1, 3), name='random_var', dtype=tf.float32) # 定义变量
# tf.SparseTensor(indices, values, dense_shape)
tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]) # 定义稀疏张量
# tf.sparse_placeholder(dtype, shape=None, name=None)
tf.sparse_placeholder(dtype=tf.float32)
```

张量的秩是它的维数，而它的形状是一个整数元组，指定了数组中每个维度的长度  。张量按NumPy数组的方式进行切片和重构 。这里提供一个进行张量操作的例子：

```python?linenums
# 定义二阶常数张量
a = tf.constant([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=tf.float32)
a_rank = tf.rank(a) # 获取张量的秩
a_shape = tf.shape(a) # 获取张量的形状
b = tf.reshape(a, [4, 2]) # 对张量进行重构
# 运行会话以显示结果
with tf.Session() as sess:
   print('constant tensor: {}'.format(sess.run(a)))
   print('the rank of tensor: {}'.format(sess.run(a_rank)))
   print('the shape of tensor: {}'.format(sess.run(a_shape)))
   print('reshaped tensor: {}'.format(sess.run(b)))
   # 对张量进行切片
   print("tensor's first column: {}".format(sess.run(a[:, 0])))
```

张量有23种数据类型，包括4类浮点实数、2类浮点复数、13类整数、逻辑、字符串和两个特殊类型，数据类型之间可以互相转换 。TensorFlow中的张量是数据流图中的单位，可以不具有值，但在图构建完毕后可以获取其中任意张量的值，该过程被称为“评估（evaluate）：

```python?linenums
constant = tf.constant([1, 2, 3]) # 定义常数张量
square = constant*constant # 操作（平方）
# 运行会话
with tf.Session() as sess:
   print(square.eval()) # “评估”操作所得常数张量的值
```

TensorFlow无法直接评估在函数内部或控制流结构内部定义的张量。如果张量取决于队列中的值，那么只有在某个项加入队列后才能评估。

### 变量（tf.Variable）

变量是可以通过操作改变取值的特殊张量  。变量必须先初始化后才可使用，低阶API中定义的变量必须明确初始化，高阶API例如Keras会自动对变量进行初始化。TensorFlow可以在tf.Session开始时一次性初始化所有变量，对自行初始化变量，在tf.Variable上运行的tf.get_variable可以在定义变量的同时指定初始化器 。这里提供两个变量初始化的例子：

```python?linenums
# 例1：使用TensorFlow的全局随机初始化器
a = tf.get_variable(name='var5', shape=[1, 2])
init = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init)
   print(a.eval())
# 例2：自行定义初始化器
# tf.get_variable(name, shape=None, dtype=None, initializer=None, trainable=None, ...)
var1 = tf.get_variable(name="zero_var", shape=[1, 2, 3], dtype=tf.float32,
 initializer=tf.zeros_initializer) # 定义全零初始化的三维变量
var2 = tf.get_variable(name="user_var", initializer=tf.constant([1, 2, 3],  dtype=tf.float32)) 
# 使用常数初始化变量，此时不指定形状shape
```

Tensorflow提供变量集合以储存不同类型的变量，默认的变量集合包括  ：

> *  本地变量：tf.GraphKeys.LOCAL_VARIABLES
> *  全局变量：tf.GraphKeys.GLOBAL_VARIABLES
> *  训练梯度变量：tf.GraphKeys.TRAINABLE_VARIABLES

用户也可以自行定义变量集合：

```python?linenums
var3 = tf.get_variable(name="local_var", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

在对变量进行共享时，可以直接引用tf.Variables，也可以使用 tf.variable_scope 进行封装：

```python?linenums
def toy_model():
   定义包含变量的操作
   var1 = tf.get_variable(name="user_var5", initializer=tf.constant([1, 2, 3], dtype=tf.float32))
   var2 = tf.get_variable(name="user_var6", initializer=tf.constant([1, 1, 1], dtype=tf.float32))
   return var1+var2
with tf.variable_scope("model") as scope:
   output1 = toy_model()
   # reuse语句后二次利用变量
   scope.reuse_variables()
   output2 = toy_model()
# 在variable_scope程序块内启用reuse
with tf.variable_scope(scope, reuse=True):
   output3 = toy_model()
```

### 数据流图（tf.Graph）和会话（tf.Session）

TensorFlow在数据流编程下运行，具体地，使用数据流图（tf.Graph）表示计算指令间的依赖关系，随后依据图创建会话（tf.Session）并运行图的各个部分 。tf.Graph包含了图结构与图集合两类相关信息，其中图结构包含图的节点（tf.Operation）和边缘（张量）对象，表示各个操作组合在一起的方式，但不规定它们的使用方式，类似于汇编代码；图集合是在tf.Graph中存储元数据集合的通用机制，即对象列表与键（tf.GraphKeys）的关联  。例如当用户创建变量时，系统将其加入变量集合，并在后续操作中使用变量集合作为默认参数  。

构建tf.Graph时将节点和边缘对象加入图中不会触发计算，图构建完成后将计算部分分流给tf.Session实现计算。tf.Session拥有物理资源，通常与Python的with代码块中使用，在离开代码块后释放资源 。在不使用with代码块的情况下创建tf.Session，应在完成会话时明确调用tf.Session.close结束进程。调用Session.run创建的中间张量会在调用结束时或结束之前释放。tf.Session.run是运行节点对象和评估张量的主要方式，tf.Session.run需要指定fetch并提供供给数据（feed）字典，用户也可以指定其它选项以监督会话的运行 。这里使用低阶API以批量梯度下降的线性回归为例展示tf.Graph的构建和tf.Session的运行：

```python?linenums
# 导入模块
import numpy as np
import tensorflow as tf
# 准备学习数据
train_X = np.random.normal(1, 5, 200) # 输入特征
train_Y = 0.5*train_X+2+np.random.normal(0, 1, 200) # 学习目标
L = len(train_X) # 样本量
# 定义学习超参数
epoch = 200 # 纪元数（使用所有学习数据一次为1纪元）
learn_rate = 0.005 # 学习速度
# 定义数据流图
temp_graph = tf.Graph()
with temp_graph.as_default():
   X = tf.placeholder(tf.float32) # 定义张量占位符
   Y = tf.placeholder(tf.float32)
   k = tf.Variable(np.random.randn(), dtype=tf.float32)
   b = tf.Variable(0, dtype=tf.float32) # 定义变量
   linear_model = k*X+b # 线性模型
   cost = tf.reduce_mean(tf.square(linear_model - Y)) # 代价函数
   optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate) # 梯度下降算法
   train_step = optimizer.minimize(cost) # 最小化代价函数
   init = tf.global_variables_initializer() # 使用变量全局初始化选项
train_curve = [] # 定义列表存储学习曲线
with tf.Session(graph=temp_graph) as sess:
   sess.run(init) # 变量初始化
   for i in range(epoch):
       sess.run(train_step, feed_dict={X: train_X, Y: train_Y}) # 运行“最小化代价函数”
       temp_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y}) # 代价函数
       train_curve.append(temp_cost) # 学习曲线
   kt_k = sess.run(k); kt_b = sess.run(b) # 运行“模型参数”
   Y_pred = sess.run(linear_model, feed_dict={X: train_X}) # 运行“模型”得到学习结果
# 绘制学习结果
ax1 = plt.subplot(1, 2, 1); ax1.set_title('Linear model fit');
ax1.plot(train_X, train_Y, 'b.'); ax1.plot(train_X, Y_pred, 'r-')
ax2 = plt.subplot(1, 2, 2); ax2.set_title('Training curve');
ax2.plot(train_curve, 'r--')
```

### 保存和恢复

TensorFlow的低阶API可以保存模型和学习得到的变量，对其进行恢复后可以无需初始化直接使用。对张量的保存和恢复使用tf.train.Saver 。里提供一个应用于变量的例子：

```python?linenums
import tensorflow as tf
# 保存变量
var = tf.get_variable("var_name", [5], initializer = tf.zeros_initializer) # 定义
saver = tf.train.Saver({"var_name": var}) # 不指定变量字典时保存所有变量
with tf.Session() as sess:
   var.initializer.run() # 变量初始化
   # 在当前路径保存变量
   saver.save(sess, "./model.ckpt")
# 读取变量
tf.reset_default_graph() # 清空所有变量
var = tf.get_variable("var_name", [5], initializer = tf.zeros_initializer)
saver = tf.train.Saver({"var_name": var}) # 使用相同的变量名
with tf.Session() as sess:
   # 读取变量（无需初始化）
   saver.restore(sess, "./model.ckpt")
```

使用检查点工具tf.python.tools.inspect_checkpoint可以查看文件中保存的张量，这里提供一个例子  ：

```python?linenums
from tensorflow.python.tools import inspect_checkpoint as chkp
# 显示所有张量（指定tensor_name=''可检索特定张量）
chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)

```

TensorFlow保存的模型使用SavedModel文件包，该文件包含是一种独立于语言（language-neutral）且可恢复的序列化格式，使较高级别的系统和工具可以创建、使用和转换 TensorFlow模型为SavedModel。tf.saved_model API可以直接与SavedModel进行交互，tf.saved_model.simple_save用于保存模型，tf.saved_model.loader.load用于导入模型。其一般用法如下：

```python?linenums
from tensorflow.python.saved_model import tag_constants
export_dir = '' # 定义保存路径
# ...（略去）定义图...
with tf.Session(graph=tf.Graph()) as sess:
   # ...（略去）运行图...
   # 保存图
   tf.saved_model.simple_save(sess, export_dir, inputs={"x": x, "y": y}, outputs={"z": z})
   tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir) # tag默认为SERVING

```

上述保存方法适用于大部分图和会话，但具体地，用户也可使用构建器（builder API）手动构建SavedModel。

## 高阶API

### Estimators

Estimators是TensorFlow自带的高阶神经网络API。Estimators封装了神经网络的训练、评估、预测、导出等操作。Estimators的特点是具有完整的可移植性，即同一个模型可以在各类终端、服务中运行并使用GPU或TPU加速而无需重新编码。Estimators模型提供分布式训练循环，包括构建图、初始化变量、加载数据、处理异常、创建检查点（checkpoint）并从故障中恢复、保存TensorBoard的摘要等。Estimators包含了预创建模型，其工作流程如下：

> * 建立数据集导入函数：可以使用TensorFlow的数据导入工具tf.data.Dataset或从NumPy数组创建数据集导入函数。
> * 定义特征列：特征列（tf.feature_column）包含了训练数据的特征名称、特征类型和输入预处理操作。
> * 调出预创建的Estimator模型：可用的模型包括基础统计学（baseline）、梯度提升决策树（boosting desicion tree）和深度神经网络的回归、分类器。调出模型后需提供输入特征列、检查点路径和有关模型参数（例如神经网络的隐含层结构）。
> * 训练和评估模型：所有预创建模型都包含train和evaluate接口用于学习和评估。

这里提供一个使用Estimator预创建的深度神经网络分类器对MNIST数据进行学习的例子：

```python?linenums
import numpy as np
import tensorflow as tf
from tensorflow import keras
# 读取google fashion图像分类数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 转化像素值为浮点数
train_images = train_images / 255.0
test_images = test_images / 255.0
# 使用NumPy数组构建数据集导入函数
train_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"pixels": train_images}, y=train_labels.astype(np.int32), shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"pixels": test_images}, y=test_labels.astype(np.int32), shuffle=False)
# 定义特征列（numeric_column为数值型）
feature_columns = [tf.feature_column.numeric_column("pixels", shape=[28, 28])]
# 定义深度学习神经网络分类器，新建文件夹estimator_test保存检查点
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, hidden_units=[128, 128], 
    optimizer=tf.train.AdamOptimizer(1e-4), n_classes=10, model_dir = './estimator_test')
classifier.train(input_fn=train_input_fn, steps=20000) # 学习
model_eval = classifier.evaluate(input_fn=test_input_fn) # 评估

```

Estimator提供“层函数（tf.layer）  ”和其它有关工具以支持用户自定义新模型，这些工具也被视为“中层API”。由于自定义完整模型过程繁琐，因此可首先使用预构建模型并完成一次训练循环，在分析结果之后尝试自定义模型。这里提供一个自定义神经网络分类器的例子：

```python?linenums
# 导入模块和数据集的步骤与前一程序示例相同
def my_model(features, labels, mode, params):
   # 仿DNNClassifier构建的自定义分类器
   # 定义输入层-隐含层-输出层
   net = tf.feature_column.input_layer(features, params['feature_columns'])
   for units in params['hidden_units']:
       net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
   logits = tf.layers.dense(net, params['n_classes'], activation=None)
   # argmax函数转化输出结果
   predicted_classes = tf.argmax(logits, 1)
   # （学习完毕后的）预测模式
   if mode == tf.estimator.ModeKeys.PREDICT:
       predictions = {'class_ids': predicted_classes[:, tf.newaxis]}
       return tf.estimator.EstimatorSpec(mode, predictions=predictions)
   # 定义损失函数
   loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
   # 计算评估指标（以分类精度为例）
   accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
   metrics = {'accuracy': accuracy}
   tf.summary.scalar('accuracy', accuracy[1])
   if mode == tf.estimator.ModeKeys.EVAL:
       # 评估模式
       return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
   else:
       # 学习模式
       assert mode == tf.estimator.ModeKeys.TRAIN
       optimizer = tf.train.AdagradOptimizer(learning_rate=0.1) # 定义优化器
       train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step()) # 优化损失函数
       return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
# 调用自定义模型，使用前一程序示例中的 1.构建数据集导入函数 和 2. 特征列
classifier = tf.estimator.Estimator(model_fn=my_model, params={
   'feature_columns': feature_columns,
   'hidden_units': [64, 64],
   'n_classes': 10})
# 学习（后续的评估/预测步骤与先前相同）
classifier.train(input_fn=train_input_fn, steps=20000)

```

Estimators的模型参数无需另外保存，在使用模型时提供检查点的路径即可调出上次学习获得的参数重新初始化模型。Estimators也支持用户自定义检查点规则。这里提供一个例子：

```python?linenums
# 每20分钟保存一次检查点/保留最新的10个检查点
my_checkpoint = tf.estimator.RunConfig(save_checkpoints_secs = 20*60, keep_checkpoint_max = 10)
# 使用新的检查点规则重新编译先前模型（保持模型结构不变）
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, hidden_units=[128, 128], 
    model_dir = './estimator_test', config=my_checkpoint)

```

除使用检查点作为对模型进行自动保存的工具外，用户也可使用低阶API将模型保存至SavedModel文件。

### Keras

Keras是一个支持TensorFlow、Thenao和Microsoft-CNTK的第三方高阶神经网络API。Keras以TensorFlow的Python API为基础提供了神经网络、尤其是深度网络的构筑模块，并将神经网络开发、训练、测试的各项操作进行封装以提升可扩展性和简化使用难度。在TensorFlow下可以直接导出Keras模块使用。这里提供一个使用tensorflow.keras构建深度神经网络分类器对MNIST数据进行学习的例子：

```python?linenums
import tensorflow as tf
from tensorflow import keras
# 读取google fashion图像分类数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 转化像素值为浮点数
train_images = train_images / 255.0
test_images = test_images / 255.0
# 构建输入层-隐含层-输出层
model = keras.Sequential([
   keras.layers.Flatten(input_shape=(28, 28)),
   keras.layers.Dense(128, activation=tf.nn.relu),
   keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 设定优化算法、损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
# 开始学习（epochs=5）
model.fit(train_images, train_labels, epochs=5)
# 模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# 预测
predictions = model.predict(test_images)
# 保存模式和模式参数
model.save_weights('./keras_test') # 在当前路径新建文件夹
model.save('my_model.h5')

```

Keras可以将模型导入Estimators以利用其完善的分布式训练循环，对上述例子，导入方式如下：

```python?linenums
# 从文件恢复模型和学习参数
model = keras.models.load_model('my_model.h5')
model.load_weights('./keras_test')
# 新建文件夹存放Estimtor检查点
est_model = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='./estimtor_test')

```

使用tensorflow.keras可以运行所有兼容Keras的代码而不损失速度，但在Python的模块管理工具中，tensorflow.keras的最新版本可能落后于Keras的官方版本。tensorflow.keras使用HDF5文件保存神经网络的权重系数。

### Eager Execution

Eager Execution是基于TensorFlow Python API的命令式编程环境，帮助用户跳过数据流编程的图操作，直接获取结果，便于TensorFlow的入门学习和模型调试，在机器学习应用中可以用于快速迭代小模型和小型数据集。Eager Execution环境只能在程序的开始，即导入tensorflow模块时启用：

```python?linenums
import tensorflow as tf
tf.enable_eager_execution()

```

Eager Execution使用Python控制流，支持标准的Python调试工具，状态对象的生命周期也由其对应的Python对象的生命周期，而不是tf.Session决定。Eager Execution支持大多数TensorFlow操作和GPU加速，但可能会使某些操作的开销增加。

### Data

tf.data是TensorFlow中进行数据管理的高阶API。在图像处理问题中，tf.data可以对输入图像进行组合或叠加随机扰动，增大神经网络的训练收益；在文字处理问题中，tf.data负责字符提取和嵌入（embedding），后者将文字转化为高维向量，是进行机器学习的重要步骤。tf.data包含两个类：tf.data.Dataset和tf.data.Iterator，Dataset自身是一系列由张量构成的组元，并包含缓存（cache）、交错读取（interleave）、预读取（prefetch）、洗牌（shuffle）、投影（map）、重复（repeat）等数据预处理方法、Iterator类似于Python的循环器，是从Dataset中提取组元的有效方式。tf.data支持从NumPy数组和TFRecord中导入数据，在字符数据处理时时，tf.data.TextLineDataset可以直接输入ASCII编码文件。

tf.data可用于构建和优化大规机器学习的输入管道（input pipline），提升TensorFlow性能。一个典型的输入管道包含三个部分：

> * 提取（Extract）：从本地或云端的数据存储点读取原始数据
> * 转化（Transform）：使用计算设备（通常为CPU）对数据进行解析和后处理，例如解压缩、洗牌（shuffling）、打包（batching）等
> * 加载（Load）：在运行机器学习算法的高性能计算设备（GPU和TPU）加载经过后处理的数据

在本地的同步操作下，当GPU/TPU进行算法迭代时，CPU处于闲置状态，而当CPU分发数据时，GPU/TPU处于闲置状态。tf.data.Dataset.prefetch在转化和加载数据时提供了预读取技术，可以实现输入管道下算法迭代和数据分发同时进行，在当前学习迭代完成时能更快地提供下一个迭代的输入数据。tf.data.Dataset.prefetch的buffer_size参数通常为预读取值的个数。

tf.data支持输入管道的并行，tf.contrib.data.parallel_interleave可以并行提取数据；映射函数tf.data.Dataset.map能够并行处理用户的指定操作。对于跨CPU并行，用户可以通过num_parallel_calls接口指定并行操作的等级。一般而言，并行等级与设备的CPU核心数相同，即在四核处理器上可定义num_parallel_calls=4。在大数据问题中，可使用tf.contrib.data.map_and_batch并行处理用户操作和分批操作。这里提供一个构建和优化输入管道的例子：

```python?linenums
import tensorflow as tf
# 使用FLAG统一管理输入管道参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_parallel_readers', 0, 'doc info')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 0, 'doc info')
tf.app.flags.DEFINE_integer('batch_size', 0, 'doc info')
tf.app.flags.DEFINE_integer('num_parallel_calls', 0, 'doc info')
tf.app.flags.DEFINE_integer('prefetch_buffer_size', 0, 'doc info')
# 自定义操作（map）
def map_fn(example):
   # 定义数据格式（图像、分类标签）
   example_fmt = {"image": tf.FixedLenFeature((), tf.string, ""),
                  "label": tf.FixedLenFeature((), tf.int64, -1)}
   # 按格式解析数据
   parsed = tf.parse_single_example(example, example_fmt)
   image = tf.image.decode_image(parsed["image"]) # 图像解码操作
   return image, parsed["label"]
# 输入函数
def input_fn(argv):
   # 列出路径的所有TFRData文件（修改路径后）
   files = tf.data.Dataset.list_files("/path/TFRData*")
   # 并行交叉读取数据
   dataset = files.apply(
      tf.contrib.data.parallel_interleave(
         tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
   dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size) # 数据洗牌
   # map和batch的并行操作
   dataset = dataset.apply(
       tf.contrib.data.map_and_batch(map_func=map_fn,
                                     batch_size=FLAGS.batch_size,
                                     num_parallel_calls=FLAGS.num_parallel_calls))
   dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size) # 数据预读取设置
   return dataset
# argv的第一个字符串为说明
tf.app.run(input_fn, argv=['pipline_params',
                       '--num_parallel_readers', '2',
                       '--shuffle_buffer_size', '50',
                       '--batch_size', '50',
                       '--num_parallel_calls, 4'
                       '--prefetch_buffer_size', '50'])

```

在输入管道的各项操作中，交叉读取、 预读取和洗牌能降低内存占用，因此具有高优先级。数据的洗牌应在重复操作前完成，为此可使用两者的组合方法tf.contrib.data.shuffle_and_repeat。

## 加速器

### CPU和GPU设备

TensorFlow支持CPU和GPU运行，在程序中设备使用字符串进行表示。CPU表示为"/cpu:0"；第一个GPU表示为"/device:GPU:0"；第二个GPU表示为"/device:GPU:1"，以此类推。如果TensorFlow指令中兼有CPU和GPU实现，当该指令分配到设备时，GPU设备有优先权。TensorFlow仅使用计算能力高于3.5的GPU设备。

在启用会话时打开log_device_placement配置选项，可以在终端查看会话中所有操作和张量所分配的设备，这里提供一个例子：

```python?linenums
# 构建数据流图.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 启用会话并设定log_device_placement=True.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
   print(sess.run(c))
# 终端中可见信息：MatMul: (MatMul): /job:localhost/replica:0/task:0/device:CPU:0…

```

默认地，TensorFlow会尽可能地使用GPU内存，最理想的情况是进程只分配可用内存的一个子集，或者仅根据进程需要增加内存使用量，为此，启用会话时可通过两个编译选项来进行GPU进程管理。

> * 内存动态分配选项allow_growth可以根据需要分配GPU内存，该选项在开启时会少量分配内存，并随着会话的运行对占用内存区域进行扩展。TensorFlow会话默认不释放内存，以避免内存碎片问题。
> * per_process_gpu_memory_fraction 选项决定每个进程所允许的GPU内存最大比例。

这里提供一个在会话中编译GPU进程选项的例子：

```python?linenums
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 开启GPU内存动态分配
config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 内存最大占用比例为40%
with tf.Session(config=config) as sess:
   # ...（略去）会话内容 ...

```

### TPU设备

张量处理器（Tensor Processing Unit, TPU）是谷歌为TensorFlow定制的专用芯片。TPU部署于谷歌的云计算平台，并作为机器学习产品开放研究和商业使用。TensorFlow的神经网络API Estimator拥有支持TPU下可运行的版本TPUEstimator。TPUEstimator可以在本地进行学习/调试，并上传谷歌云计算平台进行计算。

使用云计算TPU设备需要快速向TPU供给数据，为此可使用tf.data.Dataset API从谷歌云存储分区中构建输入管道。小数据集可使用tf.data.Dataset.cache完全加载到内存中，大数据可转化为TFRecord格式并使用tf.data.TFRecordDataset进行读取。

### 设备管理（tf.device）

TensorFlow使用tf.device对设备进行管理，tf.device的设备规范具有以下形式：

```
/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>

```

其中<JOB_NAME> 是一个字母数字字符串，并且不以数字开头。<DEVICE_TYPE> 是一种注册设备类型（例如 GPU 或 CPU）。<TASK_INDEX> 是一个非负整数，表示名为 <JOB_NAME> 的作业中的任务的索引。<DEVICE_INDEX> 是一个非负整数，表示设备索引，例如用于区分同一进程中使用的不同GPU设备。

定义变量时可以使用tf.device指定设备名称，tf.train.replica_device_setter可以对变量的设备进行自动分配，这里提供一个在不同设备定义变量和操作的例子：

```python?linenums
# 手动分配
with tf.device("/device:GPU:1"):
 var = tf.get_variable("var", [1])
# 自动分配
cluster_spec = {
   "ps": ["ps0:2222", "ps1:2222"],
   "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
 v = tf.get_variable("var", shape=[20, 20])

```

根据tf.device对变量的分配，在单一GPU的系统中，与变量有关的操作会被固定到CPU或GPU上；在多GPU的系统中，操作会在偏好设备（或多个设备同时）运行。多GPU并行处理图的节点能加快会话的运行，这里提供一个例子：

```python?linenums
c = [] # 在GPU:1和GPU:2定义张量 （运行该例子要求系统存在对应GPU设备）
for d in ['/device:GPU:1', '/device:GPU:2']:
   with tf.device(d):
       a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
       b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
       c.append(tf.matmul(a, b))
# 在CPU定义相加运算
with tf.device('/cpu:0'):
   my_sum = tf.add_n(c)
# 启用会话
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
   print(sess.run(my_sum))

```

## 优化器

### 模型优化工具

Tensorflow提供了模型优化工具（Model Optimization Toolkit）对模型的尺度、响应时间和计算开销进行优化。模型优化工具可以减少模型参数的使用量（pruning）、对模型精度进行量化（quantization）和改进模型的拓扑结构，适用于将模型部署到终端设备，或在有硬件局限时运行模型，因此有很多优化方案是TensorFlow Lite项目的一部分。其中量化能够在最小化精度损失的情况下显著减小模型尺度和缩短响应时间，并是优化深度学习模型的重要手段。这里提供一个使用使用模型优化工具的例子：

```python?linenums
import tensorflow as tf
converter = tf.contrib.lite.TocoConverter.from_saved_model(path) # 从路径导入模型
converter.post_training_quantize = True # 开启学习后量化
tflite_quantized_model = converter.convert() # 输出量化后的模型
open("quantized_model.tflite", "wb").write(tflite_quantized_model) # 写入新文件

```

### XLA

线性代数加速器（Accelerated Linear Algebra, XLA）是一个特殊的编译器，用于优化TensorFlow中的线性代数计算，其目标是优化内存使用，提升TensorFlow的运行速度和跨平台，尤其是移动终端的可移植性。

XLA工作的前端输入为“高层优化器（High Level Optimizer, HLO）”定义的数据流图，随后XLA使用多种独立于计算设备的算法优化方案对图进行分析，并将HLO计算送入后端。后端会进一步进行基于特定设备，例如GPU的优化。截至TensorFlow的1.12版本，XLA依然处于早期开发状态，暂不能提供显著的性能优化，其硬件支持包括JIT和AOT编译的x86-64 CPU、NVIDIA GPU。

## 可视化工具

TensorFlow拥有自带的可视化工具TensorBoard，TensorBoard具有展示数据流图、绘制分析图、显示附加数据等功能。开源安装的TensorFlow会自行配置TensorBoard。启动TensorBoard前需要建立模型档案，低阶API使用tf.summary构建档案，Keras包含callback方法、Estimator会自行建立档案。这里提供两个例子：

```python?linenums
# 为低层API构建档案
my_graph = tf.Graph()
with my_graph.as_default():
   # 构建数据流图
with tf.Session(graph=my_graph) as sess:
   # 会话操作   
    file_writer = tf.summary.FileWriter('/user_log_path', sess.graph) # 输出文件
# 为Keras模型构建档案
import tensorflow.keras as keras
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')
# … （略去）用户自定义模型 ...
model.fit(callbacks=[tensorboard]) # 调用fit时加载callback

```

档案建立完毕后在终端可依据档案路径运行TensorBoard主程序：

```python?linenums
tensorboard --logdir=/user_log_path

```

当终端显示TensorBoard 1.12.0 at http://your_pc_name:6006 (Press CTRL+C to quit)时，跳转至localhost:6006可使用TensorFlow界面。

## 调试程序

由于通用调试程序，例如Python的pdb很难对TensorFlow代码进行调试，因此TensorFlow团队开发了专用的调试模块TFDBG，该模块可以在学习和预测时查看会话中数据流图的内部结构和状态。TFDBG在运行时期间会拦截指令生成的错误，并向用户显示错误信息和调试说明。TFDBG使用文本交互系统curses，在不支持curses的Windows操作系统，可以下载非官方的Windows curses软件包或使用readline作为代替。使用TFDBG调试会话时，可以直接将会话进行封装，具体有如下例子：

```python?linenums
from tensorflow.python import debug as tf_debug
with tf.Session() as sess:
   sess = tf_debug.LocalCLIDebugWrapperSession(sess)
   print(sess.run(c))

```

封装容器与会话具有相同界面，因此调试时无需修改代码。封装容器在会话开始时调出命令行界面（Command Line Interface, CLI），CLI包含超过60条指令，用户可以在使用指令控制会话、检查数据流图、打印及保存张量。

TFDBG可以调试神经网络API Estimator和Keras，对Estimator，TFDBG创建调试挂钩（LocalCLIDebugHook）作为Estimator中的fit和evaluate方法下monitor的参数。对Keras，TFDBG提供Keras后端会话的封装对象，这里提供一些调试例子：

```python?linenums
# 调试Estimator
Import tensorflow as tf
from tensorflow.python import debug as tf_debug
hooks = [tf_debug.LocalCLIDebugHook()] # 创建调试挂钩
# classifier = tf.estimator. … 调用Estimator模型
classifier.fit(x, y, steps, monitors=hooks) # 调试fit
classifier.evaluate(x, y, hooks=hooks) # 调试evaluate
# 调试Keras
from keras import backend as keras_backend
# 在程序开始时打开后端会话封装
keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
# 构建Keras模型
model.fit(...)  # 使用模型学习时进入调试界面（CLI）

```

TFDBG支持远程和离线会话调试，可应用于在没有终端访问权限的远程机器（例如云计算）运行Tensorflow的场合。除CLI外，TFDBG在TensorBoard拥有拥有图形界面的调试程序插件，该插件提供了计算图检查、张量实时可视化、张量连续性和条件性断点以及将张量关联到图源代码等功能。

注：本文内容收集自网络，如有侵权，请联系删除
