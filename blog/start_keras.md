---
title: Keras快速入门教程
tags: keras,入门教程
slug: storywriter/upgrade_log
grammar_mindmap: true
renderNumberedHeading: true
grammar_code: true
grammar_decorate: true
grammar_mathjax: true

---

# 引言

Keras是一个用于构建和训练深度学习模型的高级API, 后端计算主要使用TensorFlow、Microsoft-CNTK和Theano, 由于Theano已停止开发新功能转入维护阶段, 而且目前最新版的TensorFlow已经包含了Keras模块, 所以本教程采用基于TensorFlow的Keras进行讲解, TensorFlow版本1.4.1。Keras主要用于快速原型设计，高级研究和生产中，具有三个主要优势：

### 用户友好的

Keras提供一致而简洁的API， 能够极大减少一般应用下用户的工作量，同时，Keras提供清晰和具有实践意义的bug反馈。

### 模块化和可组合的

Keras模型是通过将可配置的模块连接在一起而制定的，几乎没有限制。

### 易于扩展

很容易编写自定义模块以表达研究的新想法。创建新图层，损失函数并开发最先进的模型。

# 快速使用

快速搭建Keras示例代码：
sequential model：

```python?linenums
from keras.models import Sequential
model = Sequential()
```

可以在这个模型上加入新的层：

```python?linenums
from keras.layers import Dense
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

然后编译模型：

```python?linenums
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

或者：

```python?linenums
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

```

训练模型的时候，使用fit会自动进行batch training：

```python?linenums
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

当然也可以自己手动：

```python?linenums
model.train_on_batch(x_batch, y_batch)
```

然后使用模型进行predict：

```python?linenums
classes = model.predict(x_test, batch_size=128)
```

# 简单示例

```python?linenums
# Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPoolilng2D
from keras.utils import np_utils
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape(1, 28, 28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, epochs=10, verbose=1)

# Evaluate model on test data
score = model.evalute(X_test, Y_test, verbose=0)
```


相关链接：
[1] Keras网站：https://keras.io
[2] TensforFlow 安装：https://www.tensorflow.org/install/
[3] MNIST: https://keras.io/datasets/ 
[4] Example: https://github.com/keras-team/keras/tree/master/examples 
[5] Q&A: https://keras.io/getting-started/faq/ 
[6] Model API: https://keras.io/models/model/ 
[7] Sequential Model：https://keras.io/getting-started/sequential-model-guide/ 
[8] MNIST 模型视频： https://www.youtube.com/watch?v=L8unuZNpWw8 
https://www.youtube.com/watch?v=Ky1ku1miDow&feature=youtu.be 
https://www.youtube.com/watch?v=F1vek6ULo9w&feature=youtu.be
[9] [深度学习框架之Keras入门教程](https://blog.csdn.net/c602273091/article/details/78917069)
