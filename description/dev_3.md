# 预训练模型使用样例

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

### 
