# OCR文字识别


# 环境搭建

开发环境: Anaconda | python3.6 + tensorflow/keras/pytorch
该模型使用了 OpenCV 模块。

```python?linenums
依赖包版本需求：你可以使用 pip install 包名/ conda install 包名 安装依赖
easydict==1.7
tensorflow_gpu==1.3.0
scipy==0.18.1
numpy==1.11.1
opencv_python==3.4.0.12
Cython==0.27.3
Pillow==5.0.0
PyYAML==3.12
```

如果您没有gpu设备，演示步骤如下：
（1）将文件./ctpn/text.yml中的“USE_GPU_NMS”设置为“False” ;
（2）在文件中设置“__C.USE_GPU_NMS” ./lib/fast_rcnn/config.py为“False”;
（3）在文件./lib/fast_rcnn/nms_wrapper.py中注释掉“from lib.utils.gpu_nms import gpu_nms”;
（4）重建 setup.py：

```python?linenums
from Cython.Build import cythonize
import numpy as np
from distutils.core import setup

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
    
setup(ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"],
                            include_dirs=[numpy_include]),)

（a）执行导出CFLAGS = -I/home/zhao181/ProGram1/anaconda2/lib/python2.7/site-packages/numpy/core/include 
你应该使用自己的numpy路径。

（b）cd xxx/text-detection-ctpn-master/lib/utils 
和execute：python setup.py build

（c）将.so文件从“build”目录复制到xxx/text-detection-ctpn-master/lib/utils。

（5）cd xxx/text-detection-ctpn-master并执行：python ./ctpn/demo.py

顺便说一下，我使用
Anaconda2-4.2.0-Linux-x86_64.sh和tensorflow-1.3.0-cp27-cp27mu-manylinux1_x86_64.whl（cpu）在ubuntu 16.04下运行。
```

如果您有一个gpu设备，请按以下方式构建库

```python?linenums
cd lib / utils
chmod + x make.sh
./make.sh
```

Github地址：https://github.com/eragonruan/text-detection-ctpn
我们使用 Classify(vgg16) 来检测文本方向，使用CTPN(CNN+RNN) 来检测文本区域，使用 CRNN(CNN+GRU/LSTM+CTC) 来进行 EndToEnd的文本识别。

> 1.文本方向检测网络-Classify(vgg16)
> 2.文本区域检测网络-CTPN(CNN+RNN)
> 3.EndToEnd文本识别网络-CRNN(CNN+GRU/LSTM+CTC)

基于图像分类，在VGG16模型的基础上，训练0、90、180、270度检测的分类模型（考虑到文本在纸张上出现的情况）。代码参考angle/predict.py文件，训练图片8000张，准确率88.23%
关于 OCR 端到端识别:CRNN网络请查看 https://blog.csdn.net/wsp_1138886114/article/details/82555728
你可以运行demo.py 写入测试图片的路径来测试识别效果，
如果想要显示ctpn的结果，修改文件./ctpn/ctpn/other.py 的draw_boxes函数的最后部分，cv2.inwrite(‘dest_path’,img)，如此，可以得到ctpn检测的文字区域框以及图像的ocr识别结果。

# 训练网络

工程项目目录

```python?linenums
"""
root
.
├── ctpn
|   ├── __init__.py
|   ├── demo.py
|   ├── demo_pb.py
|   ├── generate_pb.py
|   ├── text.yml
|   └── train_net.py
├── data
|   ├── demo
|   ├── oriented_results
|   ├── results
|   ├── ctpn.pb
|   └── results
└── lib
    ├── __pycache__
    ├── datasets
    ├── fast_rcnn
    ├── networks
    ├── prepare_training_data
    ├── roi_data_layer
    ├── rpn_msr
    ├── text_connector
    ├── utils
    └── __init__.py

"""
```

1. 对ctpn进行训练

> * 定位到路径–./ctpn/ctpn/train_net.py
> * 预训练的vgg网络路径VGG_imagenet.npy将预训练权重下载下来，pretrained_model指向该路径即可,此外整个模型的预训练权重checkpoint
> * ctpn数据集还是百度云数据集下载完成并解压后，将.ctpn/lib/datasets/pascal_voc.py 文件中的pascal_voc 类中的参数self.devkit_path指向数据集的路径即可

2. 对crnn进行训练

> * keras版本 ./train/keras_train/train_batch.py model_path--指向预训练权重位置MODEL_PATH---指向模型训练保存的位置keras模型预训练权重
> * pythorch版本./train/pytorch-train/crnn_main.py

```python?linenums
parser.add_argument( ‘–crnn’, help=“path to crnn (to continue training)”, default=预训练权重的路径)
parser.add_argument( ‘–experiment’, help=‘Where to store samples and models’, default=模型训练的权重保存位置,这个自己指定)
```

对于纯文字的识别结果还行，感觉可以在crnn网络在加以改进，现在的crnn中的cnn有点浅，并且rnn层为单层双向+attention，目前正在针对这个地方进行改动，使用迁移学习，以restnet为特征提取层，使用多层双向动态rnn+attention+ctc的机制，将模型加深。
关于训练需求准备

> * 首先，下载预先训练的VGG网络模型并将其放入data / pretrain / VGG_imagenet.npy中。
> * 其次，准备纸上提到的培训数据，或者您可以下载我从谷歌驱动器或百度云准备的数据。或者您可以按照以下步骤准备自己的数据。
> * 根据您的数据集修改prepare_training_data / split_label.py中的路径和gt_path。并运行：

```shell?linenums
cd lib/prepare_training_data
python split_label.py

数据集准备好后：
python ToVoc.py

将准备好的训练数据转换为voc格式。它将生成一个名为TEXTVOC的文件夹。将此文件夹移动到数据/然后运行：
cd ../../data
ln -s TEXTVOC VOCdevkit2007

```

训练
python ./ctpn/train_net.py

> * 你可以在ctpn / text.yml中修改一些超级参数，或者只使用我设置的参数。
> * 我在检查站提供的模型在GTX1070上训练了50k iters。
> * 如果你正在使用cuda nms，它每次约需0.2秒。因此完成50k迭代需要大约2.5小时。






本文来源：

> * [基于深度学习（端到端）的OCR文字识别](https://blog.csdn.net/wsp_1138886114/article/details/83864582)
> * [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)

