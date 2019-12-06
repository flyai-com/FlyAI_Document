# 常见问题

**Q：如何获得奖金？**

A：超过项目设置的最低分，根据公式计算，就可以获得奖金。

**Q：比赛使用什么框架？**

A：比赛支持常用的机器学习和深度学习框架，比如TensorFlow，PyTorch，Keras，Scikit-learn等。

**Q：怎么参加比赛，需不需要提交csv文件？**

A：FlyAI竞赛平台无需提交csv文件，在网页上点击报名，下载项目，使用你熟练的框架，修改`main.py`中的网络结构，和`processor.py`中的数据处理。使用FlyAI提供的命令提交，就可以参加比赛了。

**Q：比赛排行榜分数怎么得到的？**

A：参加项目竞赛必须实现 `model.py` 中的`predict_all`方法。系统通过该方法，调用模型得出评分。

**Q：平台机器什么配置？**

A：目前每个训练独占一块P40显卡，显存24G。

**Q：本地数据集在哪？**

A：运行 `flyai.exe test` or `./flyai test` 命令之后会下载100条数据到项目的data目录下，也可以本地使用ide运行 `main.py` 下载数据

**Q：FlyAI自带的Python环境在哪,会不会覆盖本地环境？**

A：FlyAI不会覆盖本地现有的Python环境。

* windows用户:

  C:Users{你计算机用户名}.flyaienvpython.exe

  C:Users{你计算机用户名}.flyaienvScriptspip.exe

* mac和linux用户:

  /Users/{你计算机用户名}/.flyai/env/bin/python3.6

  /Users/{你计算机用户名}/.flyai/env/bin/pip

**Q：FAI训练积分不够用怎么办？**

A：目前GPU免费使用，可以进入到：[我的积分](https://www.flyai.com/personal_score)，通过签到和分享等途径获得大量积分。

**Q：离线训练代码不符合规范问题**

A：`main.py`中必须使用`args.EPOCHS`和`args.BATCH`。

**Q：项目什么时候开始，持续多长时间？**

A：网站上能看见的项目就是已经开始的，项目会一直存在，随时可以提交。

**Q：排行榜和获奖问题**

A：目前可能获得奖金的项目是审核之后才能上榜的，每天会审核一次。通过审核之后，奖金才能发送到账户。

**Q：全量数据集怎么查看，数据集支不支持下载？**

A：暂时不支持查看和下载，如果需要，可以进入数据集来源链接查看。运行 `flyai.exe train -e=xx -b=xx` or `./flyai train -e=xx -b=xx` 命令之后会提交代码到服务器上使用全量数据集训练。

**Q：from flyai.dataset import Dataset 报错、No module name "flyai"**

A：先找到ide中使用的Python对应的pip的位置。

* windows用户：pip所在路径pip.exe install -i [https://pypi.flyai.com/simple](https://pypi.flyai.com/simple) flyai
* mac和linux用户：pip所在路径/pip install -i [https://pypi.flyai.com/simple](https://pypi.flyai.com/simple) flyai
* 其他 No module name "xxxx"问题 也可以参考上面 

**Q：预训练模型如何加载使用**

FlyAI提供预训练模型链接地址：[https://www.flyai.com/models](https://www.flyai.com/models)

keras 加载预训练模型样例：

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

pytorch加载预训练模型样例：

```python
import torchvision
from flyai.utils import remote_helper
path=remote_helper.get_remote_date("https://www.flyai.com/m/resnet50-19c8e357.pth")
model = torchvision.models.resnet50(pretrained = False)
# model = torchvision.models.resnet50(pretrained = True)
# 这行代码与上面等同，只不过一个是调用FlyAI提供的预训练模型地址，一个是外网的地址
model.load_state_dict(torch.load(path)
model.fc = nn.Linear(2048,200) # 将其中的层直接替换为我们需要的层即可
```

tensorflow加载bert预训练模型样例：

```python
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
path = remote_helper.get_remote_date('https://www.flyai.com/m/multi_cased_L-12_H-768_A-12.zip')
# 参数
lr = 0.0006  # 学习率
rnn_type = 'lstm'
rnn_size = 64
layer_num = 3
numClasses = 2
keep_prob = 1.0
# 使用本地路径
data_root = os.path.join(os.path.curdir, 'data/input/model/multi_cased_L-12_H-768_A-12')
bert_config_file = os.path.join(data_root, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
bert_vocab_file = os.path.join(data_root, 'vocab.txt')
# 导入数据
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
input_y = tf.placeholder(tf.float32, shape=[None, numClasses], name="input_y")
# 初始化BERT
model = modeling.BertModel(
  config=bert_config,
  is_training=False,
  input_ids=input_ids,
  input_mask=input_mask,
  token_type_ids=segment_ids,
  use_one_hot_embeddings=False)
# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
# 获取最后一层
output_layer_seq = model.get_sequence_output()  # 这个获取每个token的output
tf.identity(model.get_pooled_output(), name='output_layer_pooled')
```

**Q：如何将flyai框架里的dataset转换成pytorch的dataloader? pytorch 数据增强如何应用在flyai的dataset上? 训练集和验证集的预处理方式可以不一样么?**

首先这里提供一种将FlyAI框架里的dataset转换成pytorch的dataloader的方式，同时实现了数据增强，及在训练和验证集上使用不同的增强方式。

```python
from flyai.dataset import Dataset

class FlyAIDataset(Dataset):
  def __init__(self, x_dict, y_dict, train_flag=True):
      self.images = [x['image_path'] for x in x_dict]
      self.labels = [y['label'] for y in y_dict]
      if train_flag:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.RandomHorizontalFlip(), # 随机水平翻转
                  transforms.RandomVerticalFlip(), # 随机竖直翻转
                  transforms.RandomRotation(30), #（-30，+30）之间随机旋转
                  transforms.ToTensor(), #转成tensor[0, 255] -> [0.0,1.0]
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      else:
          self.transform = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  def __len__(self):
      return len(self.images)

  def __getitem__(self, index):
      path = os.path.join(DATA_PATH, self.images[index])
      image = Image.open(path)
      img = self.transform(image)
      label = self.labels[index]
      return img, label

data = Dataset()
x_train, y_train, x_val, y_val = data.get_all_data() # 获取全量数据
# x_train: [{'image_path': 'img/10479.jpg'}, {'image_path': 'img/14607.jpg'},   {'image_path': 'img/851.jpg'}...]
# y_train: [{'label': 39}, {'label': 4}, {'label': 3}...]
train_dataset = FlyAIDataset(x_train, y_train)
val_dataset = FlyAIDataset(x_val, y_val, train_flag=False)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.BATCH)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.BATCH)
```

**Q：关于线上评估问题**

经常有用户遇到线上评估出错的问题，所以这里提供在线下可以调用**model.py中predict\_all函数的方法**

```python
from flyai.dataset import Dataset
from model import Model
import sys

dataset = Dataset()
model = Model(dataset)

# 调用predict_all的方法
x_test = [{'image_path': 'img/10479.jpg'}, {'image_path': 'img/14607.jpg'}]
y_test = [{'label': 39}, {'label': 4}]
preds = model.predict_all(x_test)
labels = [i['label'] for i in y_test]

# 调用predict的方法
img_path = 'img/851.jpg'
p = model.predict(image_path=img_path)
print(p)
```

**Q：线上环境安装报错问题**

这个大家需要根据自己代码需要的环境在**requirement.txt**文件中写上自己需要安装的包的版本，格式如下：

```text
flyai
numpy
Pillow
tensorflow-gpu==1.13.0
```

**Q：linux下提交训练，epoch和batch参数不生效**

epoch和batch不生效的几点原因：

1&gt; 当提交训练的代码和样例代码重复度比较高的时候，我们在线上是不调GPU进行训练的

2&gt; 确定在**main.py**函数中是否设置了这个参数

```python
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
```

然后在模型训练的时候是否正确使用了这个参数

```python
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
for step in range(dataset.get_step()): # (train_len / args.BATCH)* args.EPOCHS
    # 接下来网络训练...
```

