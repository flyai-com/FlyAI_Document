####  

### FlyAI算法项目开发流程

> FlyAI的python库版本需大于等于0.6.6

#### 一、数据处理

1.下载数据，把不同的数据自己处理成csv的形式

> 数据格式如图所示，文件使用相对路径，用左斜线分割路径"/"
>
> 其他数据放入csv中，如果标注数据非常大，也可以放入标注文件路径

![image-20200316174531993](https://dataset.flyai.com/image-1.png)

![image-20200316174544704](https://dataset.flyai.com/image-2.png)



2.生成dataset.json文件

```python
from flyai.utils.projcet_helper import Project
procjet=Project()
project.generate_dataset("数据集ID---大驼峰命名，每个单词首字母大写，不要加下滑线", "数据集的名字", 
                         "数据集的描述", "数据集的来源", 1(int类型，百分比，本地训练时使用的比例),
                         ['数据集的标签 如图像分类', '推荐算法'])
```

3.dataset.json文件在项目中生成之后，打开自己修改"data_info"、"model"和"data_path"字段

>data_info 是对数据集csv文件的描述
>model 是模型的输入和输出定义
>data_path是整理成csv之后的数据集所在的路径

4.使用FlyAI脚本划分数据集

```python
from flyai.cc import dataset_split, upload_dataset_from_json
dataset_split("dataset.json文件路径",
               columns=["指定csv中是文件路径的那些列，如image_path,label_path"],
               helpers=["除csv之外，其他的辅助文件，没有辅助文件直接把该字段删除即可，如词向量，词表 直接填写该文件所在路径即可"], 	
               public_size=100, validation_size=100)
# public_size 是本地训练集大小,如果赋值大于1，则表示该数据集划分多少条数据。
# 如果小于1，则表示使用总数据量的百分比
# validation_size 为验证集大小，规则同public_size
```

划分成功的数据会生成到当前数据的output文件夹下如图所示，表示数据集划分成功

![image-20200316175853613](https://dataset.flyai.com/image-3.png)



5.把划分好的数据集上传到FlyAI服务器。

```python
 from flyai.cc import dataset_split, upload_dataset_from_json
 upload_dataset_from_json("dataset.json的文件路径")
```

> 数据集上传之后，才能在代码中下载使用

```python
data_helper = DataHelper()
data_helper.download_from_ids(DataId)
```



#### 二、使用FlyAI模版代码编写

1.生成项目模版

> dataset.json需要与该代码文件同级才能生成

```python
from flyai.utils.projcet_helper import Project
procjet=Project()
project.generate_code("使用的算法 如CNN", "使用的框架 如Tensorflow")
```

生成的结构

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

##### 具体文件详细解释

* main.py

  > **每个项目的样例代码中已做简单实现，可供查考。**
  >
  > 程序入口，编写算法，训练模型的文件。在该文件中实现自己的算法。

  ```python
  # -*- coding: utf-8 -*-
  import argparse
  
  from flyai.data_helper import DataHelper
  from flyai.framework import FlyAI
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
  parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
  args = parser.parse_args()
  
  #继承 flyai.framework 中的 FlyAI模板类
  class Main(FlyAI):
   def download_data(self):
       '''
       下载数据
       :return:
       '''
       data_helper = DataHelper()
  		 # 根据数据ID下载训练数据
       data_helper.download_from_ids("data_id xxxx")
       # 二选一或者根据app.json的配置下载文件
       data_helper.download_from_json()
         
   def deal_with_data(self):
       '''
       处理数据，如果没有可以不实现。
       :return:
       '''
       pass
  
   def train(self):
       '''
       训练模型，必须实现此方法
       :return:
       '''
       pass
  
  
  if __name__ == '__main__':
   main = Main()
   main.download_data()
   main.deal_with_data()
   main.train()
  
  
  ```

  

* `prediction.py`

  > **每个项目的样例代码中已做简单实现，可供查考。**
  >
  > 训练好模型之后可以继承`flyai.model.base`包中的`base`重写下面三个方法实现模型的保存、验证和使用。

  ```python
  # -*- coding: utf-8 -*
  #使用FlyAI提供的类
  from flyai.framework import FlyAI
  
  #继承FlyAI实现加载模型和预测方法
  class Prediction(FlyAI):
   def load_model(self):
       '''
       模型初始化，必须在此方法中加载模型
       '''
       pass
  
   def predict(self, **input_data):
       '''
       模型预测返回结果
       :param input: 评估传入样例，是key-value字典类型 例如：
       {"user_id": 31031, "post_id": 3530, "create_post_user_id": 27617, "post_text": "心情棒棒哒"}
       :return: 模型预测成功返回，也是字典类型 例如：{"label": 0}
       '''
       return 返回的例子 {"label": 0}
  ```

  

* `path.py`

  >数据、日志、模型的公共路径

  ```python
  # -*- coding: utf-8 -*
  import sys
  
  import os
  
  #数据下载路径
  DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
  #模型保存路径
  MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
  
  
  ```

  



#### 三、提交GPU训练

提交训练之前，分清楚项目是`FlyAI的竞赛样例`，还是`公司项目`

1.FlyAI竞赛样例项目 

> 数据集划分规则，public_size为0.2～0.6, validation_size为0.2
>
> 使用flyai.exe train 提交训练，模型上传和评估，FlyAI竞赛系统会自动进行

2.公司项目使用AI训练平台提交训练

> 数据集划分规则，public_size为100，validation_size为100
>
> 训练平台不会自动上传模型和评估，需要自己实现模型评估和上传

##### 提交训练

```python
from flyai.aa import PAI

pai = PAI("公司分配的用户名", "密码")

# 提交训练 
# 可选参数
# run="main.py", 运行命令 后面可自己加参数  如 main.py -batch=xx 
# 使用的docker镜像，可以自己定义镜像
# docker_image="reg.xxwolo.com/flyai/flyai-gpu:openpai-v2.2"
pai.train("数据集id", "项目的git地址：https://git.xxwolo.com/flyai/simple/postrecommendation.git")

# 提交之后查看训练日志
https://flyai.com/training_result/trainid_xxxx
#查看训练任务
pai.get_job_list()

# 通过ID停止训练任务
pai.stop_train("tain70b8d8132dd9bcfb5733a67f")
```

##### 模型上传和评估，

> 服务器上训练好的模型需要上传

```python
from flyai.cc import upload_model
upload_model("服务器上模型路径+名称",True or Flase 是否覆盖上传)
```

> 本地下载训练好的模型

```python
 from flyai.cc import download
 download("上传的模型名称即可，不要添加路径")
```

