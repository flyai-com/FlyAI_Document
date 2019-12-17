# 语音分类任务



# 介绍

我们需要实现10种语音的分类：冷气机，汽车喇叭，儿童玩耍，狗吠声，钻孔，发动机空转，枪射击，手持式凿岩机，警笛，街头音乐，每个录音长度约为4s，被放在10个fold文件中。
我们采用keras（可以简单的认为keras是前端，tensorflow是后端，类似于tensorflow是个库，我们使用keras调用它的api）实现模型搭建，使用librosa（Librosa是一个用于音频、音乐分析、处理的python工具包）来处理语音。 

### 第一步，导入这几个库即可

Git URL: 

```python?linenums
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
```

### 第二步，读取csv文件

```python?linenums
data = pd.read_csv('metadata/UrbanSound8K.csv')
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 3 ]
valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
```

### 第三步，读入wav文件

```python?linenums
from tqdm import tnrange, tqdm_notebook

D=[]

for row in tqdm_notebook(valid_data.itertuples()): 
    print(row.path)
    print(row.classID)
    y1, sr1 = librosa.load("audio/" + row.path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    if ps.shape != (128, 128): 
            continue
    D.append( (ps, row.classID) )
```

### 第四步，划分训练集和测试集，前7000个为训练集，7000以后为数据集

```python?linenums
dataset = D
random.shuffle(dataset)

train = dataset[:7000]
test = dataset[7000:]

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])


y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))
```

### 第五步，搭建模型

```python?linenums
model = Sequential()
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
```

### 第六步，填入数据

```python?linenums
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'])

model.fit(
    x=X_train, 
    y=y_train,
    epochs=12,
    batch_size=128,
    validation_data= (X_test, y_test))

score = model.evaluate(
    x=X_test,
    y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

如果没有问题的话，模型就开始训练了，时间不会很长，因为模型搭的很简单。 
我的准确率是在67%，有些偏低，不过后续可以继续改进。 
这是我的github地址[github](https://github.com/yaokaishile/speech12)，有兴趣的童鞋可以点个star。



本文来源：

> * [语音分类任务（基于UrbanSound8K数据集）](https://blog.csdn.net/c2c2c2aa/article/details/81543549)

