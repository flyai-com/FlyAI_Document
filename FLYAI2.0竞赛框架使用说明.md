

### $NAME

### [![GPL LICENSE](https://badgen.net/badge/License/GPL/green)](https://www.gnu.org/licenses/gpl-3.0.zh-cn.html) [![GPL LICENSE](https://badgen.net/badge/Supported/TensorFlow,Keras,PyTorch/green?list=1)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Python/3.+/green)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Platform/Windows,macOS,Linux/green?list=1)](https://flyai.com)

### [项目官方网址](https://www.flyai.com/d/$DATAID)

> $DESCRIPTION

***

#### 1.参赛流程

	> 本地使用的FlyAI Python库版本需要大于等于0.6.4

* 第一步：参赛选手从[FlyAI官网](https://www.flyai.com)选择比赛报名，可在线查看代码并下载代码

  > 下载的项目中不包含数据集，运行main.py会自动下载调试数据集
  >
  > 本地调试根据不同数据集会提供60%～100%数据，全量数据提交到GPU后会自动更新替换

* 第二步：本地代码调试

  > 本地配置Python3.5以上的运行环境，并安装项目运行所需的Python依赖包
  > app.json是项目的配置文件
  >
  > 在main.py中编写神经网络，没有框架限制
  >
  > 在prediction.py测试模型是否评估成功
  >
  > main.py中需在class Main(FlyAI) 类中实现自己的的训练过程

* 第三步：提交到GPU训练，保存模型

  >本地调试完成之后，提交代码到GPU，在全量数据上训练模型，保存最优模型。
  >
  >提交GPU的方式有：FlyAI客户端、FlyAI脚本命令、网站在线提交。

* 第四步：评估模型，获取奖金，实时提现

  >GPU训练完成后，会调用prediction.py中的predict方法进行评估，并给出最后得分
  >
  >高分的参赛选手，可实时获取奖金，通过微信提现

比赛遇到问题不要着急可以添加FlyAI小助手微信，小姐姐在线解答您的问题。

<img src="https://static.flyai.com/flyai_dir4.png" alt="FlyAI小助手微信二维码" style="zoom:50%;" />

#### 2.样例项目结构说明

![项目目录](https://static.flyai.com/project2.png)



* **`flyai.exe/flyai`**

  >Windows用户双击flyai.exe,启动FlyAI客户端，本地调试项目
  >
  >MAC和Linux用户在终端执行flyai脚本，本地调试项目
  >
  >[点击查看使用详情](https://doc.flyai.com/description/dev_1.html)

  参赛选手还可以使用自己电脑上的Python环境，安装项目依赖，运行main.py来调试。

* **`使用jupyter调试.ipynb`**

  >比赛项目可以使用jupyter本地调试和提交
  >
  >flyai.exe支持一键配置并启动jupyter lab本地调试和提交

* **`app.json`**

  > 项目的说明描述文件，用来查看项目，不需要修改

* **`main.py`**

  > 项目运行主文件，在该文件中编写神经网络代码

* **`prediction.py`**

  >模型本本评估调试文件

* **`path.py`**

  >数据、日志、模型的公共路径

* **`requirements.txt`**

  > 项目中用到的Python依赖
  >
  > 引入新依赖时需要准确填写
  >
  > 不填写版本号将默认安装最新版

参赛选手还可以查看[样例项目代码详细说明](https://doc.flyai.com/description/dev_2.html)

#### 3.FlyAI的Python库文档

为了便于数据的管理和处理，我们官方封装了FlyAI库

FlyAI的库和其它Python库的安装方法一样，使用PIP工具安装

> windows用户：pip所在路径pip.exe install -i https://pypi.flyai.com/simple flyai==0.6.6
>
> mac和linux用户：pip所在路径/pip install -i https://pypi.flyai.com/simple flyai==0.6.4

FlyAI库的主要用途是，读取比赛数据。

使用方式如下：

```python
#引入flyai数据下载类，DataHelper
from flyai.data_helper import DataHelper
#初始化类对象
data_helper = DataHelper()
# 根据数据ID下载训练数据
data_helper.download_from_ids("data_id xxxx")
# 二选一或者根据app.json的配置下载文件
data_helper.download_from_json()
```

##### 新的数据集会下载到 `.data/input/data_id_xxx/ `目录下

***

#### <a name="div3">样例项目代码详细说明</a>

* `main.py`

  > **每个项目的样例代码中已做简单实现，可供查考。**
  >
  > 程序入口，编写算法，训练模型的文件。在该文件中实现自己的算法。
  >
  > ```python
  > # -*- coding: utf-8 -*-
  > import argparse
  > 
  > from flyai.data_helper import DataHelper
  > from flyai.framework import FlyAI
  > 
  > '''
  > 样例代码仅供参考学习，可以自己修改实现逻辑。
  > 模版项目下载支持（PyTorch、Tensorflow、Keras、MXNET等
  > 第一次使用请看项目中的：FLYAI2.0框架项目详细文档.html
  > 使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
  > 学习资料可查看文档中心：https://doc.flyai.com/
  > 常见问题：https://doc.flyai.com/question.html
  > 遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
  > '''
  > 
  > parser = argparse.ArgumentParser()
  > parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
  > parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
  > args = parser.parse_args()
  > 
  > 
  > 
  > 
  > #继承 flyai.framework 中的 FlyAI模板类
  > class Main(FlyAI):
  > '''
  > 项目中必须继承FlyAI类，否则线上运行会报错。
  > '''
  > def download_data(self):
  >   '''
  >   下载数据
  >   :return:
  >   '''
  >   data_helper = DataHelper()
  > 		 # 根据数据ID下载训练数据
  >   data_helper.download_from_ids("data_id xxxx")
  >   # 二选一或者根据app.json的配置下载文件
  >   data_helper.download_from_json()
  >     
  > def deal_with_data(self):
  >   '''
  >   处理数据，如果没有可以不实现。
  >   :return:
  >   '''
  >   pass
  > 
  > def train(self):
  >   '''
  >   训练模型，必须实现此方法
  >   :return:
  >   '''
  >   pass
  > 
  > 
  > if __name__ == '__main__':
  > main = Main()
  > main.download_data()
  > main.deal_with_data()
  > main.train()
  > 
  > ```
  >
  > 

* `prediction.py`

  > **每个项目的样例代码中已做简单实现，可供查考。**
  >
  > 训练好模型之后可以继承`flyai.model.base`包中的`base`重写下面三个方法实现模型的保存、验证和使用。
  >
  > ```python
  > # -*- coding: utf-8 -*
  > #使用FlyAI提供的类
  > from flyai.framework import FlyAI
  > 
  > #继承FlyAI实现加载模型和预测方法
  > class Prediction(FlyAI):
  >  def load_model(self):
  >      '''
  >      模型初始化，必须在此方法中加载模型
  >      '''
  >      pass
  > 
  >  def predict(self, **input_data):
  >      '''
  >      模型预测返回结果
  >      :param input: 评估传入样例，是key-value字典类型 例如：{"user_id": 31031, "post_id": 3530, "create_post_user_id": 27617, "post_text": "心情棒棒哒"}
  >      :return: 模型预测成功返回，也是字典类型 例如：{"label": 0}
  >      '''
  >      return 返回的例子 {"label": 0}
  > ```

* `path.py`

  >数据、日志、模型的公共路径
  >
  >```python
  ># -*- coding: utf-8 -*
  >import sys
  >
  >import os
  >
  >#数据下载路径
  >DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
  >#模型保存路径
  >MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
  >
  >```
  >
  >

***

#### 4.预训练模型的使用

比赛可以使用FlyAI网站上公开的与训练模型

> 模型查找地址：https://www.flyai.com/models

在网页中找到自己想要用的模型，之后点击“复制使用”按钮。

![image-20200109150415456](https://static.flyai.com/flyai_dir3.png)

粘贴之后显示如下：

```python
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
# 下载到项目中的data/input/文件夹，默认会自动解压，具体文件路径可以下之后查看使用
path = remote_helper.get_remote_data('https://www.flyai.com/m/bert-base-uncased.tar.gz')
```

具体使用请查看[预训练模型使用样例](#div4)，其它高级使用方式，请选手自行查找。

##### flyai.utils的remote_helper方法

```python
def get_remote_data(remote_name, unzip=True):
    """
    下载FlyAI网站上的预训练模型
    :param remote_name:模型的下载地址 
    :param unzip: 是否解压
    :return: 返回没解压的路径，具体解压位置请下载之后查看
    """
    return remote.get_remote_data(remote_name, unzip)
```



****

#### <a name="div1">参赛选手常见问题</a>

>**Q：使用自己的Python环境，遇到No module name "flyai"**
>
>A：先找到使用的Python对应的pip.exe的位置
>
>（自己电脑上可能有多个Python和pip，安装目标不要弄错。）
>
>- windows用户在终端执行：pip所在路径pip.exe install -i https://pypi.flyai.com/simple flyai==0.6.4
>- mac和linux用户在终端执行：pip所在路径/pip install -i https://pypi.flyai.com/simple flyai==0.6.4
>- 其他 No module name "xxxx"问题 也可以参考上面
>
>**Q：FlyAI自带的Python环境在哪,会不会覆盖本地环境？**
>
>A：FlyAI不会覆盖本地现有的Python环境。
>
>- windows用户:
>
> C://Users//{你计算机用户名}//.flyai//env//python.exe
>
> C://Users//{你计算机用户名}///.flyai//env//Scriptspip.exe
>
>- mac和linux用户:
>
> /Users/{你计算机用户名}/.flyai/env/bin/python3.6
>
> /Users/{你计算机用户名}/.flyai/env/bin/pip

其它更多常见问题，请访问文档中心查看:[常见问题](https://doc.flyai.com/question.html)

***

#### <a name="div2">FlyAI本地调试代码指南</a>

#### 方式一：Windows客户端调试

##### 1. 下载项目并解压

##### 2.进入到项目目录下，双击执行flyai.exe程序

> 第一次使用需要使用微信扫码登录
>
> 杀毒软件可能会误报，点击信任该程序即可

##### 3.本地开发调试

> 运行flyai.exe程序，点击"本地调试"按钮，输入循环次数和数据量，点击运行即可调用main.py
>
> 如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖
>
> 如果使用本地IDE开发，需要安装“flyai”依赖并导入项目，运行main.py 

##### 4.下载本地测试数据

> 运行flyai.exe程序，点击"下载数据"按钮，程序会下载60%～100%数据

##### 4.提交到GPU训练

项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，否则线上运行会找不到包。

不填写版本号将默认安装最新版

> 运行flyai.exe程序，点击"提交到GPU"按钮，输入循环次数和数据量，点击运行即可提交到GPU训练。

返回sucess状态，代表提交离线训练成功

训练结束会以微信和邮件的形式发送结果通知

#### 方式二：使用Jupyter调试

运行flyai.exe程序，扫码登录之后

点击"使用jupyter调试"按钮，一键打开jupyter lab 操作界面

##### 1.本地运行

在jupter中运行 run main.py 命令即可在本地训练调试代码

如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

##### 2.提交到GPU训练

项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，否则线上运行会找不到包。

不填写版本号将默认安装最新版

在jupyter环境下运行  ! flyai.exe train -e=10 -b=32云端GPU免费训练

> 返回sucess状态，代表提交离线训练成功
>
> 训练结束会以微信和邮件的形式发送结果通知

#### 方式三：windows命令行调试

##### 1. 下载项目并解压

##### 2. 打开运行，输入cmd，打开终端

> Win+R 输入cmd

##### 3. 使用终端进入到项目的根目录下

首先进入到项目对应的磁盘中，然后执行

> cd path\to\project
>
> Windows用户使用 flyai.exe

##### 4. 本地开发调试

执行下列命令本地安装环境并调试（第一次使用需要使用微信扫码登录）

> flyai.exe test

执行test命令，会自动下载100条测试数据到项目下

如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

如果使用本地IDE开发，可以自行安装 requirements.txt 中的依赖，运行 main.py 即可

##### 5.提交到GPU训练

项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，否则线上运行会找不到包。

不填写版本号将默认安装最新版

在终端下执行

> flyai.exe train

返回sucess状态，代表提交离线训练成功

训练结束会以微信和邮件的形式发送结果通知

完整训练设置执行代码示例：

> flyai.exe train -b=32 -e=10

通过执行训练命令，整个数据集循环10次，每次训练读取的数据量为 32 。

***

#### <a name="div_n">Mac和Linux调试</a>

#####方式一：命令行调试

##### 1. 下载项目并解压

##### 2. 使用终端进入到项目的根目录下

> cd /path/to/project
>
> Mac和Linux用户使用 ./flyai 脚本文件

##### 3. 初始化环境并登录

授权flyai脚本

> chmod +x ./flyai

##### 4. 本地开发调试

执行下列命令本地安装环境并调试（第一次使用需要使用微信扫码登录）

> ./flyai  test   注意:如果pip安装中出现 permission denied 错误，需使用sudo运行

执行test命令，会自动下载100条测试数据到项目下

如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

如果使用本地IDE开发，可以自行安装 requirements.txt 中的依赖，运行 main.py 即可

##### 5.提交到GPU训练

项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，否则线上运行会找不到包。

不填写版本号将默认安装最新版

在终端下执行

> ./flyai train 

返回sucess状态，代表提交离线训练成功

训练结束会以微信和邮件的形式发送结果通知

完整训练设置执行代码示例：

> ./flyai train -b=32 -e=10

通过执行训练命令，整个数据集循环10次，每次训练读取的数据量为 32 。

##### 方式二：使用Jupyter调试

在终端执行命令 ./flyai ide 打开调试环境，扫码登录成功之后

##### 1.本地运行

在jupter中运行 run main.py 命令即可在本地训练调试代码

如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

##### 2.提交到GPU训练

项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，否则线上运行会找不到包。

不填写版本号将默认安装最新版

在jupyter环境下运行  ! ./flyai train -e=10 -b=32  将代码提交到云端GPU免费训练

> 返回sucess状态，代表提交离线训练成功
>
> 训练结束会以微信和邮件的形式发送结果通知

### 设置自己的Python环境

##### Windows用户

> flyai.exe path=xxx 可以设置自己的Python路径
>
> flyai.exe path=flyai 恢复系统默认Pyton路径

##### Mac/linux用户

> ./flyai path=xxx 可以设置自己的Python路径
>
> ./flyai path=flyai 恢复系统默认Pyton路径

***

#### <a name="div4">预训练模型使用样例</a>

##### Keras预训练模型使用样例：

```python
from keras.applications import densenet
from flyai.utils import remote_helper
path=remote_helper.get_remote_date("https://www.flyai.com/m/v0.8|densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5")
densenet_notop = densenet.DenseNet169(include_top=False, weights=None)
densenet_notop.load_weights(path)
# densenet_notop = densenet.DenseNet169(include_top=False， weights='imagenet')
# 这行代码与上面等同，只不过一个是调用FlyAI提供的预训练模型地址，一个是外网的地址
x = densenet_notop.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)
model = Model(inputs=densenet_notop.input, outputs=predictions)
model.compile(...)
model.fit_generator(...)
```

##### PyTorch预训练模型使用样例：

```python
import torchvision
from flyai.utils import remote_helper
path=remote_helper.get_remote_date("https://www.flyai.com/m/resnet50-19c8e357.pth")
model = torchvision.models.resnet50(pretrained = False)
# model = torchvision.models.resnet50(pretrained = True)
# 这行代码与上面等同，只不过一个是调用FlyAI提供的预训练模型地址，一个是外网的地址
model.load_state_dict(torch.load(path)
# 将其中的层直接替换为我们需要的层即可                      
model.fc = nn.Linear(2048,200)
```

##### Tensorflow加载Bert预训练模型样例：

```python
import tensorflow as tf
import bert.modeling as modeling
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper

path = remote_helper.get_remote_date('https://www.flyai.com/m/uncased_L-12_H-768_A-12.zip')
print('path:', path)
data_root = os.path.splitext(path)[0]
print('data_root:', data_root)

# 解析link解压后的路径
data_root = os.path.splitext(path)[0]
# 【注意】使用改路径前首先确认是否和预训练model下载解压路径是否一致
print('data_root:', data_root) 
# 使用当前路径
# 预训练model路径存放地址和link解析路径不一致时使用下面方法直接指定】
# data_root = os.path.join(os.path.curdir, 'data/input/XXXX/XXXXX')
bert_config_file = os.path.join(data_root, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
bert_vocab_file = os.path.join(data_root, 'vocab.txt')
```

### [FlyAI全球人工智能专业开发平台，一站式服务平台](https://flyai.com)

**扫描下方二维码，及时获取FlyAI最新消息，抢先体验最新功能。**



[![GPL LICENSE](https://www.flyai.com/images/coding.png)](https://flyai.com)

