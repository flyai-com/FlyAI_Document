# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
下载模版之后需要把当前样例项目的app.yaml复制过去哦～
第一次使用请看项目中的：FLYAI项目详细文档.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''


'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
进入Dataset类中可查看方法说明
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
net = Net().to(device)

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0
for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    '''
    实现自己的模型保存逻辑
    '''
    model.save_model(net, MODEL_PATH, overwrite=True)
    print(str(step + 1) + "/" + str(dataset.get_step()))
