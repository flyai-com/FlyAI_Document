### 简化版

#### 第一步:安装FlyAI

在自己的Python环境中安装flyai库，版本大于等于0.6.8。

#### 第二步:上传数据集和模型

FlyAI会给每个用户分配免费的在线数据集空间，用来保存你的数据集和模型。

```python
# 上传自己的数据集
upload_data(data_file="D:/我的数据集地址/dataset.zip", overwrite=True)
# 上传自己想用的预训练模型
upload_data(data_file="D:/我的预训练模型/model.pkl", overwrite=True)
```

#### 第三步:提交到GPU训练

```python
# 在项目下创建submit.py文件，然后运行即可提交。
submit(train_name="", cmd='python main.py')
# 保存GPU上优秀的模型到数据网盘
sava_train_model(model_file="./data/output/你的服务器上模型的名字", overwrite=False)
```
