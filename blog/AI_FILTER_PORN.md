# FlyAI AI图像鉴黄项目

## 1、项目简介

本文主要介绍使用PyTorch框架通过构建SENet网络实现AI图像鉴黄服务, 对图片是否涉黄进行准确分类。
在互联网大数据时代，每天都有数以亿计的图片产生和传播，给我们生活带来便利的同时，也给互联网的监管带来了挑战，例如色情图片的传播。如何快速准确的甄别出黄色图片是大多数涉及图片内容的应用程序迫切需要解决的问题，而基于AI的图像鉴黄服务是一个非常合适的解决方案。在降低成本、减少审查时间负担的同时，能够提供一个洁净的网络环境。

## 2、数据集来源

该数据集来源于github上的一个开源项目 nsfw_data_scrapper。数据集网址链接：[nsfw_data_scrapper](https://github.com/alexkimxyz/nsfw_data_scrapper)
为了便于训练表示，我们清洗此数据集后，将图像所属类别和标签做了一一对应，关系如下所示：

```
drawings, hentai, neutral, porn, sexy
0,1,2,3,4
```

在[ FlyAI竞赛平台上 ](https://www.flyai.com) 提供了超详细的[参考代码](https://www.flyai.com)，我们可以通过参加[图像鉴黄练习赛](https://www.flyai.com)进行进一步学习和优化。主要部分代码实现如下：

## 3、代码实现

### 3.1、算法流程及实现

算法流程主要分为以下三个部分进行介绍：

1. 数据加载
2. 构建网络
3. 模型训练

#### 数据加载

在FlyAI的项目中封装了Dataset类，可以实现对数据的一些基本操作，比如加载批量训练数据*next_train_batch()*和校验数据*next_validation_batch()*、获取全量数据*get_all_data()*、获取训练集数据量*get_train_length()* 和获取校验集数据量*get_validation_length()*等。具体使用方法如下：

```python
# 引入Dataset类
from flyai.dataset import Dataset

# 加载数据辅助类
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=32, type=int, help="val batch size")
args = parser.parse_args()

'''
flyai库中提供的数据处理方法
传入整个数据训练轮数，每批次批大小
'''
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
model = Model(dataset)

# dataset.get_step()返回训练总步长（args.EPOCHS个epoches的总batch数）
# 加载processor.py中处理好的数据，按batch加载train和val数据
train_img, train_label = dataset.next_train_batch()
val_img, val_label = dataset.next_validation_batch()
print('Load data done!')

'''
在给出的样例中，我们选择加载全部原始的未经处理的数据，再统一处理。大家可以按照自己的使用需求灵活选择flyai的Dataset类的数据加载方式
'''
'''
首先加载全部原始数据，需要注意，此时得到的x_train, x_val是图片的路径，y_train, y_val是对应的类序号（0到4之间的一个数字）
'''
x_train, y_train, x_val, y_val = flyai_dataset.get_all_data()
# 然后通过ListDataset类来处理数据
train_data = ListDataset(x_train, y_train, data_path, augment=True)
# 然后生成dataloader
train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_data.collate_fn,
        pin_memory=True,
        num_workers=4)

#验证数据处理方式同上
'''
ListDataset类具体实现如下：
'''
class ListDataset(Dataset):
    def __init__(self, list_img, list_label, data_path, img_size=224, augment=True):
        self.img_files = list_img
        self.label_files = list_label
        self.data_path = data_path
        self.img_size = img_size
        self.augment = augment
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        '''
        此时的img_files是字典格式{'image_path': 'images/xxx.jpg'}，key值与app.yaml中的input:->columns一致，根据自己获取数据的方式做相应修改
        ''' 
        img_dict = self.img_files[index % len(self.img_files)]
        img_path = img_dict["image_path"].rstrip()

        # Extract image
        img = transforms.ToTensor()(Image.open(self.data_path + img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)

        # ---------
        #  Label
        # ---------
        '''
        此时的label是字典格式{'label': 0}，key值与app.yaml中的output:->columns:一致，根据自己获取数据的方式做相应修改
		'''
        label_dict = self.label_files[index % len(self.img_files)]
        label = label_dict["label"]

        # Tensor ->PIL Image
        img = transforms.ToPILImage()(img)

        # Apply augmentations
        if self.augment:
            img, label = data_aug(img, label)

        # convert to PyTorch tensor
        img = transforms.ToTensor()(img)

        # resize img to input shape
        img = resize(img, config.img_size)

        return img, label

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files)
'''
ListDataset类中用到的 pad_to_square()，data_aug()，及resize()方法具体可在样例代码中的utils.py文件中查看。
此时，我们就可以在迭代过程中通过下面的方式来获取每代的训练数据
'''
for iter, (input, target) in enumerate(train_dataloader):
    ......
```

#### 构建网络

这里我们采用2017年ImageNet图像分类竞赛中的冠军方案的网络架构SENet，此架构能够方便地嵌入到大多数之前提出的网络架构中，通过增加通道之间的联系来提高网络性能。我们采用基于50层的resnet网络来搭建一个50层的seresnet50网络。
嵌入到resnet网络中的senet基本模块在se_module.py定义：

```python
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

将上述模块融入resnet前我们需要定义一个SEBottleneck类，具体如下：

```python
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

下面我们将构建一个50层的seresnet50网络

```python
def se_resnet50(num_classes=5, pretrained=False):
    """
	Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
	''' 
	加载预训练模型，我们的网站(https://www.flyai.com/)提供了丰富的预训练模型，根据自己需要进行选择。将下面括号内的url替换为在我们网站上复制的地址
	'''
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(           "url"))
    return model
```

到此，我们的网络基本上已经构建完成，接下来就是利用准备好的数据对Model进行训练。

#### 模型训练

我们的损失函数和优化器，学习率等定义如下：

```python
# 在训练中我们使用交叉熵损失函数，定义如下：
criterion = nn.CrossEntropyLoss().to(device)

# 采用AdamOptimizer为网络优化器，定义如下：
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)

#在训练过程中采用学习率衰减策略，定义如下：
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

在训练集中每个epoch完成后，我们在验证集上执行一次迭代，评估函数定义如下：

```python
def evaluate(val_loader, model, criterion, epoch):
    # define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    # progress bar
    val_progressor = ProgressBar(
        mode="Val  ",
        epoch=epoch,
        total_epoch=config.epochs,
        model_name=config.model_name,
        total=len(val_loader)
    )
    # switch to evaluate mode and confirm model has been transfered to cuda
    model.to(device)
    model.eval()
    # 评估时关闭网络梯度
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input.to(device))
            target = Variable(torch.from_numpy(np.array(target)).long().to(device))

            # compute output
            output = model(input)
            #output = torch.LongTensor(output)

            loss = criterion(output, target)

            # measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg, top1.avg]
```

现在可以迭代进行完整的网络训练了：

```python
    for epoch in range(start_epoch, opt.EPOCHS):
        train_progressor = ProgressBar(
            mode="Train",
            epoch=epoch,
            total_epoch=opt.EPOCHS,
            model_name=config.model_name,
            total=len(train_dataloader)
        )

        for iter, (input, target) in enumerate(train_dataloader):
            # switch to continue train process
            train_progressor.current = iter
            model.train()
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target)).long()).to(device)
            output = model(input)
            # print("output:", output.shape)
            loss = criterion(output, target)

            precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()

        scheduler.step(epoch)
        train_progressor.done()
        # evaluate
        lr = get_learning_rate(optimizer)
        # evaluate every half epoch
        print("launch a evaluation!")
        valid_loss = evaluate(val_dataloader, model, criterion, epoch)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1], best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_precision1":best_precision1,
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "valid_loss":valid_loss,
        }, is_best, fold)
```

### 3.2、最终结果

最终Model的结果通过平均精度mAP进行评估。下面是该项目的可运行[完整代码链接](https://www.flyai.com)，具体细节可查看完整代码。

#### 参考链接：

-  [图像鉴黄练习赛](https://www.flyai.com)
-  [图像鉴黄练习赛代码](https://www.flyai.com)
