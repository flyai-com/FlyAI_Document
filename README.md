

### [FlyAI竞赛平台](https://www.flyai.com)

### [![GPL LICENSE](https://badgen.net/badge/License/GPL/green)](https://www.gnu.org/licenses/gpl-3.0.zh-cn.html) [![GPL LICENSE](https://badgen.net/badge/Supported/TensorFlow,Keras,PyTorch/green?list=1)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Python/3.+/green)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Platform/Windows,macOS,Linux/green?list=1)](https://flyai.com)

#### 1.参赛流程

* 第一步：参赛选手从[FlyAI官网](https://www.flyai.com)选择比赛报名，可在线查看代码并下载代码

  > 下载的项目中不包含数据集，运行main.py会自动下载调试数据集
  >
  > 本地调试只有少量的数据，全量数据提交到GPU后会自动更新替换

* 第二步：本地代码调试

  > 本地配置Python3.5以上的运行环境，并安装项目运行所需的Python依赖包
  >
  > 在main.py中编写神经网络，在processor.py中处理数据
  >
  > 使用model.py测试模型是否保存成功
  >
  > 使用predict.py测试模型是否评估成功
  >
  > main.py中必须使用args.EPOCHS和args.BATCHl来读取数据

* 第三步：提交到GPU训练，保存模型

  >本地调试完成之后，提交代码到GPU，在全量数据上训练模型，保存最优模型。
  >
  >提交GPU的方式有：FlyAI客户端、FlyAI脚本命令、网站在线提交。

* 第四步：评估模型，获取奖金，实时提现

  >GPU训练完成后，会调用model.py中的predict_all方法评估，并给出最后得分
  >
  >高分的参赛选手，可实时获取奖金，通过微信提现

参赛选手还可以查看[样例项目代码详细说明](#div3)

比赛遇到问题不要着急可以添加FlyAI小助手微信，小姐姐在线解答您的问题。

<img src="https://static.flyai.com/flyai_dir4.png" alt="FlyAI小助手微信二维码" style="zoom:50%;" />

#### 2.样例项目结构说明

<img src="https://static.flyai.com/flyai_dir_2.png" alt="FlyAI项目目录" style="zoom:50%;" />



* **`flyai.exe/flyai`**

  >Windows用户双击flyai.exe,启动FlyAI客户端，本地调试项目
  >
  >MAC和Linux用户在终端执行flyai脚本，本地调试项目
  >
  >[点击查看使用详情](#div2)

  参赛选手还可以使用自己电脑上的Python环境，安装项目依赖，运行main.py来调试。

* **`使用jupyter调试.ipynb`**

  >比赛项目可以使用jupyter本地调试和提交
  >
  >flyai.exe支持一键配置并启动jupyter lab本地调试和提交

* **`app.yaml`**

  > 项目的配置文件，默认不需要修改

* **`main.py`**

  > 项目运行主文件，在该文件中编写神经网络代码

* **`net.py`**

  > 使用PyTorch的选手可以在该文件中编写神经网络，并main.py中使用。
  >
  > 使用其它框架的选手可以忽略该文件

* **`processor.py`**

  > 数据输入(input)、输出(output)统一处理文件
  >
  > FlyAI统一并简化了数据处理流程，参赛选手需要遵循统一的数据处理方式

* **`model.py`**

  >包含模型保存和评估方法
  >
  >排行榜分数调用该文件中predict_all方法获取，请参赛选手本地调试确保没问题

* **`predict.py`**

  >模型本本评估调试文件

* **`path.py`**

  >数据、日志、模型的公共路径

* **`requirements.txt`**

  > 项目中用到的Python依赖
  >
  > 引入新依赖时需要准确填写

参赛选手还可以查看[样例项目代码详细说明](#div3)

#### 3.FlyAI的Python库文档

为了便于数据的管理和处理，我们官方封装了FlyAI库

FlyAI的库和其它Python库的安装方法一样，使用PIP工具安装

> windows用户：pip所在路径pip.exe install -i https://pypi.flyai.com/simple flyai
>
> mac和linux用户：pip所在路径/pip install -i https://pypi.flyai.com/simple flyai

FlyAI库的主要用途是，处理并读取比赛数据，底层使用多线程实现，提高数据读取效率。

使用方式如下：

```python
#引入flyai数据处理类Dataset
from flyai.dataset import Dataset
#初始化类对象，并设置整个数据集循环次数和批次大小
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
#获取一批训练数据
x_train, y_train = dataset.next_train_batch()
#获取一批验证数据
x_val, y_val = dataset.next_validation_batch()
```

***

##### flyai.dataset文件详细文档

```python
class Dataset:
    def __init__(self, epochs=5, batch=32, val_batch=32):
        """
        :param epochs: 训练的轮次，最大不超过100
        :param batch: 训练的批次大小，太大会导致显存不足
        :param val_batch: 验证的批次大小
        """
        self.lib = Lib(epochs, batch, val_batch)

    def get_step(self):
        """
        根据dataset传入的epochs和batch，计算出来的训练总次数。
        :return: 返回训练总次数
        """
        return self.lib.get_step()

    def get_train_length(self):
        """
        获取训练集总数量，本地调用返回的是100条，在GPU上调用返回全部数据集数量。
        :return: 返回训练集总数量
        """
        return self.lib.get_train_length()

    def get_validation_length(self):
        """
        获取验证集总数量，本地调用返回的是100条，在GPU上调用返回全部数据集数量。
        :return: 返回验证集总数量
        """
        return self.lib.get_validation_length()

    def next_train_batch(self):
        """
        获取一批训练数据，返回数据的数量是dataset中batch的大小。
        :return: x_train,y_train
        """
        return self.lib.next_train_batch()

    def next_validation_batch(self):
        """
        获取一批验证数据，返回数据的数量是dataset中val_batch的大小。
        :return: x_val,y_val
        """
        return self.lib.next_validation_batch()

    def next_batch(self, size=32, test_size=32, test_data=True):
        """
        获取一批训练和验证数据，可以自己设置返回的大小。
        :return:x_train,y_train,x_val,y_val
        """
        return self.lib.next_batch(size, test_size, test_data)

    def get_all_processor_data(self):
        """
        获取所有在processor.py中，通过input_x方法处理过的数据
        :return:x_train,y_train,x_val,y_val
        """
        return self.lib.get_all_processor_data()

    def get_all_data(self):
        """
        获取所有原始数据
        :return:x_train,y_train,x_val,y_val
        """
        return self.lib.get_all_data()

    def get_all_validation_data(self):
        """
        获取所有在processor.py中，通过input_x方法处理过的验证集数据
        :return:x_val,y_val
        """
        return self.lib.get_all_validation_data()

```



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
>- windows用户在终端执行：pip所在路径pip.exe install -i https://pypi.flyai.com/simple flyai
>- mac和linux用户在终端执行：pip所在路径/pip install -i https://pypi.flyai.com/simple flyai
>- 其他 No module name "xxxx"问题 也可以参考上面
>
>**Q：FlyAI自带的Python环境在哪,会不会覆盖本地环境？**
>
>A：FlyAI不会覆盖本地现有的Python环境。
>
>- windows用户:
>
>  C://Users//{你计算机用户名}//.flyai//env//python.exe
>
>  C://Users//{你计算机用户名}///.flyai//env//Scriptspip.exe
>
>- mac和linux用户:
>
>  /Users/{你计算机用户名}/.flyai/env/bin/python3.6
>
>  /Users/{你计算机用户名}/.flyai/env/bin/pip

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

> 运行flyai.exe程序，点击"下载数据"按钮，程序会下载100条调试数据

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

#### <a name="div3">样例项目代码详细说明</a>

* `main.py`

  > **样例代码中已做简单实现，可供查考。**
  >
  > 程序入口，编写算法，训练模型的文件。在该文件中实现自己的算法。
  >
  > 通过`dataset.py`中的`next_batch`方法获取训练和测试数据。
  >
  > ```python
  > '''
  > Flyai库中的提供的数据处理方法
  > 传入整个数据训练多少轮，每批次批大小
  > '''
  > dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
  > #获取训练数据
  > x_train, y_train = dataset.next_train_batch()
  > #获取验证数据
  > x_val, y_val = dataset.next_validation_batch()
  > ```
  >
  > 通过`model.py`中的`save_model`方法保存模型
  >
  > ```python
  > # 模型操作辅助类
  > model = Model(dataset)
  > model.save_model(YOU_NET)
  > ```
  >
  > **如果使用`PyTorch`框架，需要在`net.py`文件中实现网络。其它用法同上。**

* `processor.py`

  > **样例代码中已做简单实现，可供查考。**
  >
  > 处理数据的输入输出文件，把通过csv文件返回的数据，处理成能让程序识别、训练的矩阵。
  >
  > 可以自己定义输入输出的方法名，在`app.yaml`中声明即可。
  >
  > ```python
  >  def input_x(self, $INPUT_PARAMS):
  >      '''
  >  	参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
  >  	和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
  >  	该方法字段与app.yaml中的input:->columns:对应
  >  	'''
  >      pass
  > 	
  >  def output_x(self, $INPUT_PARAMS):
  >       '''
  >  	参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
  >  	和dataset.next_validation_batch()多次调用。
  >  	该方法字段与app.yaml中的input:->columns:对应
  >  	'''
  >      pass
  >  
  >  def input_y(self, $OUTPUT_PARAMS):
  >      '''
  >      参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
  >  	和dataset.next_validation_batch()多次调用。
  >  	该方法字段与app.yaml中的output:->columns:对应
  >      '''
  >      pass
  >  
  >  def output_y(self, data):
  >      '''
  >      输出的结果，会被dataset.to_categorys(data)调用
  >      :param data: 预测返回的数据
  >      :return: 返回预测的标签
  >      '''
  >      pass
  > 
  > ```

  ##### 

* `model.py`

  > **样例代码中已做简单实现，可供查考。**
  >
  > 训练好模型之后可以继承`flyai.model.base`包中的`base`重写下面三个方法实现模型的保存、验证和使用。
  >
  > ```python
  > def predict(self, **data):
  >      '''
  >      	使用模型
  >    		:param data: 模型的输入的一个或多个参数
  >      	:return:
  >      '''
  >      pass
  > 
  >  def predict_all(self, datas):
  >      '''
  >      （必须实现的方法）评估模型，对训练的好的模型进行打分
  >    		:param datas: 验证集上的随机数据，类型为list
  >      	:return outputs: 返回调用模型评估之后的list数据
  >      '''
  >      pass
  > 
  >  def save_model(self, network, path=MODEL_PATH, name=MODEL_NAME, overwrite=False):
  >      '''
  >      保存模型
  >      :param network: 训练模型的网络
  >      :param path: 要保存模型的路径
  >      :param name: 要保存模型的名字
  >      :param overwrite: 是否覆盖当前模型
  >      :return:
  >      '''
  >      self.check(path, overwrite)
  > 
  > ```
  
  predict_all的参数格式
  
  ```python
  from flyai.dataset import Dataset
  from model import Model
  import sys
  
  dataset = Dataset()
  model = Model(dataset)
  
  # predict_all的参数是多个字典组成的列表类型的数据集结构
  x_test = [{'image_path': 'img/10479.jpg'}, {'image_path': 'img/14607.jpg'}]
  y_test = [{'label': 39}, {'label': 4}]
  preds = model.predict_all(x_test)
  labels = [i['label'] for i in y_test]
  print(labels)
  # predict是单个字典模式
  img_path = 'img/851.jpg'
  p = model.predict(image_path=img_path)
  print(p)
  ```
  
  

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



