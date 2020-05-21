### FlyAI-GPU使用说明

***

#### 第一步:安装FlyAI

* 在自己的Python环境中安装flyai库，版本大于等于0.6.9。

* FlyAI库的安装方法与PyTorch、Keras等其他Python库安装方式一样。
* 注意：自己电脑上可能有多个Python和pip，安装目录不要弄错哦。

#### 第二步:上传数据集和模型

FlyAI会给每个用户分配免费的在线数据空间，用来保存你的数据集和模型。

* 如图，显示为默认数据空间提供的MnistSimple所创建的文件夹，您也可以在本地自行创建并使用Upload_data方法上传到您的数据空间中。
![默认数据空间文件展示](https://static.flyai.com/dataspace.png)
* 选择【data】文件夹，可以看到官方已准备好的MNIST.zip 数据集文件；
* 你可以点击`复制`将加载该数据集的方法代码块插入到工程文件中，方便实现；
![复制数据集使用代码](https://static.flyai.com/dataspace2.png)

以下是官方提供的完整实现代码，请作参考：
```python
# data_file:数据集的路径
# overwrite:模型名称相同的时候再上传是否覆盖，True会覆盖，False系统会重新命名
# 下载直接使用文件名称下载即可
# dir_name:文件夹名称，可以创建目录，用做斜线划分，目录不要有中文和特殊字符
# 例如:"/data" "/mydata/mnist"
upload_data("D:/data/MNIST.zip", overwrite=True)
# 上传之后在服务器上使用文件名下载数据集
# 服务器上数据下载地址为 ./MNIST.zip  decompression为True会自动解压
download("MNIST.zip", decompression=True)

# 或者设置路径上传数据，会自动在您的数据盘中创建路径
upload_data("D:/data/MNIST.zip", overwrite=True, dir_name="/data")
# 服务器上数据下载地址为 ./data/MNIST.zip  decompression为True会自动解压
download("/data/MNIST.zip", decompression=True)
```

#### 第三步:提交到GPU训练

```python
# train_name: 提交训练的名字,推荐使用英文，不要带特殊字符
# code_path: 提交训练的代码位置，不写就是当前代码目录，也可以上传zip文件
# cmd: 在服务器上要执行的命令，多个命令可以用 && 拼接
# 如：pip install -i https://pypi.flyai.com/simple keras && python train.py -e=10 -b=30 -lr=0.0003
# 会把当前submit所在的代码目录提交，cmd可以自己编写，GPU上使用python开头即可
submit("train_mnist", cmd="python train.py")

# 另一种提交方式，提交代码压缩包,目前支持zip格式的压缩包,代码会自动解压到运行目录下
submit("train_mnist", "D:/xxxxx.zip", cmd="python train.py")
```

***

##### 其他使用说明

1.保存线上训练的模型

```python
# 上传自己的数据集
# model_file 模型在服务器上的路径加名字
# overwrite 是否覆盖上传
# dir_name 模型保存在数据盘中的目录
sava_train_model(model_file="./data/output/你的服务器上模型的名字", dir_name="/model", overwrite=False)
```

2.FlyAI安装加速

> 网络不好的同学FlyAI还提供国内镜像源，使用方式如下：
>
> pip install -i https://pypi.flyai.com/simple flyai==0.6.9

3.使用自己的Python环境，遇到No module name "flyai"

先找到使用的Python对应的pip.exe的位置

> 自己电脑上可能有多个Python和pip，安装目标不要弄错。
>
> windows用户在终端执行：pip所在路径pip.exe install -i https://pypi.flyai.com/simple flyai==0.6.9
>
> mac和linux用户在终端执行：pip所在路径/pip install -i https://pypi.flyai.com/simple flyai==0.6.9
>
> 其他 No module name "xxxx"问题 也可以参考上面

4.GPU环境配置

* Python为3.6版本
* cuda为10.1

> 如有其他环境需求可以添加小姐姐微信：flyaixzs，我们为您配置专属训练环境。



使用过程遇到问题不要着急，可以添加FlyAI小助手微信，小姐姐在线解答您的问题。

<img src="https://static.flyai.com/flyai_dir4.png" alt="FlyAI小助手微信二维码" style="zoom:50%;" />




