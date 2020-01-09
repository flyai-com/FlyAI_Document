# 样例项目代码详细说明

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
  > def input_x(self, $INPUT_PARAMS):
  >   '''
  > 	参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
  > 	和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
  > 	该方法字段与app.yaml中的input:->columns:对应
  > 	'''
  >   pass
  > 	
  > def output_x(self, $INPUT_PARAMS):
  >    '''
  > 	参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
  > 	和dataset.next_validation_batch()多次调用。
  > 	该方法字段与app.yaml中的input:->columns:对应
  > 	'''
  >   pass
  > 
  > def input_y(self, $OUTPUT_PARAMS):
  >   '''
  >   参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
  > 	和dataset.next_validation_batch()多次调用。
  > 	该方法字段与app.yaml中的output:->columns:对应
  >   '''
  >   pass
  > 
  > def output_y(self, data):
  >   '''
  >   输出的结果，会被dataset.to_categorys(data)调用
  >   :param data: 预测返回的数据
  >   :return: 返回预测的标签
  >   '''
  >   pass
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
  >   '''
  >   	使用模型
  > 		:param data: 模型的输入的一个或多个参数
  >   	:return:
  >   '''
  >   pass
  > 
  > def predict_all(self, datas):
  >   '''
  >   （必须实现的方法）评估模型，对训练的好的模型进行打分
  > 		:param datas: 验证集上的随机数据，类型为list
  >   	:return outputs: 返回调用模型评估之后的list数据
  >   '''
  >   pass
  > 
  > def save_model(self, network, path=MODEL_PATH, name=MODEL_NAME, overwrite=False):
  >   '''
  >   保存模型
  >   :param network: 训练模型的网络
  >   :param path: 要保存模型的路径
  >   :param name: 要保存模型的名字
  >   :param overwrite: 是否覆盖当前模型
  >   :return:
  >   '''
  >   self.check(path, overwrite)
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

#### 
