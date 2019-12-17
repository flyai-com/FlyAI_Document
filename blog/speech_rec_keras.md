# keras实现语音识别


# 介绍

市面上语音识别技术原理已经有很多很多了，然而很多程序员兄弟们想研究的时候却看的头大，一堆的什么转mfcc，然后获取音素啥的，对于非专业音频研究者或非科班出生的程序员来说，完全跟天书一样。
最近在研究相关的实现，并且学习了keras和tensorflow等。用keras做了几个项目之后，开始着手研究语音识别的功能，在网上下载了一下语音的训练文件，已上传到了[百度云盘](https://pan.baidu.com/s/1Au85kI_oeDjode2hWumUvQ)

# 具体实现

拿到一个语音文件之后需要先转mfcc，这个操作很简单，不需要什么高深的内功。用python写一段函数专门用来获取语音文件的fmcc值。

```python?linenums
def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

    ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的
    #【因为训练文件全部是1秒钟长度，16000帧的，所以这里需要把每个语音文件的长度处理成一样的】
    data = list(np.array(waveData[0]))
    # print(len(data))
    while len(data)>16000:
        del data[len(waveData[0])-1]
        del data[0]
    # print(len(data))
    while len(data)<16000:
        data.append(0)
    # print(len(data))

    data=np.array(data)
    # 平方之后，开平方，取正数，值的范围在  0-1  之间
    data = data ** 2
    data = data ** 0.5
    return data
```

参数为单个文件在磁盘的位置，mfcc是一堆的正数和负数组成的数组。为了在训练的时候避免损失函数应为负数导致输出结果相差太大，需要把原始的mfcc全部转为正数，直接平方后在开方就是正值了。
我们可以把每个音频的mfcc值当做对应的特征向量，然后进行训练，我这里为了测试速度，取了seven 和 stop 两个语音类别来进行训练和识别，每个大概2700多个文件。并且分别从两个文件夹中剪切出来100个当做测试集，并每样拿出5个当做后面的试验集。

训练之前需要先读取数据创建数据集和标签集：

```python?linenums
# 加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    wavs=[]
    labels=[] # labels 和 testlabels 这里面存的值都是对应标签的下标，下标对应的名字在 labsInd 和 testlabsInd 中
    testwavs=[]   
    testlabels=[]
    labsInd=[]      ## 训练集标签的名字   0：seven   1：stop
    testlabsInd=[]  ## 测试集标签的名字   0：seven   1：stop
    # 现在为了测试方便和快速直接写死，后面需要改成自动扫描文件夹和标签的形式
    #加载seven训练集
    path="D:\\wav\\seven\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path+i)
        # print(waveData)
        wavs.append(waveData)
        if ("seven" in labsInd)==False:
            labsInd.append("seven")
        labels.append(labsInd.index("seven"))
    #加载stop训练集
    path="D:\\wav\\stop\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("stop" in labsInd)==False:
            labsInd.append("stop")
        labels.append(labsInd.index("stop"))
    #加载seven测试集
    path="D:\\wav\\test1\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("seven" in testlabsInd)==False:
            testlabsInd.append("seven")
        testlabels.append(testlabsInd.index("seven"))
    #加载stop测试集
    path="D:\\wav\\test2\\"
    files = os.listdir(path)
    for i in files:
        # print(i)
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("stop" in testlabsInd)==False:
            testlabsInd.append("stop")
        testlabels.append(testlabsInd.index("stop"))

    wavs=np.array(wavs)
    labels=np.array(labels)
    testwavs=np.array(testwavs)
    testlabels=np.array(testlabels)
    return (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd)
```

拿到数据集之后就可以开始进行神经网络的训练了，keras提供了很多封装好的可以直接使用的神经网络，我们先建立神经网络模型：

```python?linenums
# 构建一个4层的模型
model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(16000,))) # 音频为16000帧的数据，这里的维度就是16000，激活函数直接用常用的relu
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 因为只有两个类别的语音，最后输出应该就是2个分类的结果
# [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#  validation_data为验证集
model.fit(wavs, labels, batch_size=124, epochs=5, verbose=1, validation_data=(testwavs, testlabels)) ## 进行5轮训练，每个批次124个

# 开始评估模型效果 # verbose=0为不输出日志信息
score = model.evaluate(testwavs, testlabels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) # 准确度
```

最后保存模型到文件：

```python?linenums
model.save('asr_model_weights.h5') # 保存训练模型
```

现在训练的模型已经有了，我们开始使用trunk中的文件进行试验。
先加载之前训练的模型：

```python?linenums
model = load_model('asr_model_weights.h5') # 加载训练模型
```

然后获得当前需要试验的文件的mfcc。并且将数据封装成和训练时一样的维度。并且使用模型的predict函数输出结果：

```python?linenums
wavs=[]
wavs.append(get_wav_mfcc("D:\\wav\\trunk\\2c.wav")) # 使用某一个文件
X=np.array(wavs)
print(X.shape)
result=model.predict(X[0:1])[0] # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
print("识别结果",result)
```

结果输出：
识别结果 [0.10070908, 0.8992909]

可以看出结果是一个2个数的数组，里面返回的对应类别相似度，也就是说哪一个下标的值最大，就跟那个下标对应的标签最相似。
之前训练的时候，标签的集是：[seven , stop]

```python?linenums
#  因为在训练的时候，标签集的名字 为：  0：seven   1：stop    0 和 1 是下标
name = ["seven","stop"] # 创建一个跟训练时一样的标签集
ind=0 # 结果中最大的一个数
for i in range(len(result)):
    if result[i] > result[ind]:
        ind=1
print("识别的语音结果是：",name[ind])
```

识别结果是：stop

我们把试验文件换成 1b.wav

```python?linenums
wavs.append(get_wav_mfcc("D:\\wav\\trunk\\1b.wav"))
```

结果输出：
识别结果 [0.9871939, 0.0128061]
语音识别结果是：seven

本文相关的代码已上传[github](https://github.com/BenShuai/kerasTfPoj/tree/master/kerasTfPoj/ASR)


本文来源：

> * [python+keras实现语音识别](https://blog.csdn.net/sunshuai_coder/article/details/83658625)

