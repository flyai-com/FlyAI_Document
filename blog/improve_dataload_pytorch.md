# 优化Pytorch的数据加载


# 背景

在利用深度学习解决图像问题时，影响训练效率最大的有时候是GPU，有时候也可能是CPU和你的磁盘。
很多设计不当的任务，在训练神经网络的时候，大部分时间都是在从磁盘中读取数据，而不是做 Backpropagation 。这种症状的体现是使用 Nividia-smi 查看 GPU 使用率时，Memory-Usage 占用率很高，但是 GPU-Util 时常为 0% ，如下图所示：
![nvidia-smi](./images/nvidia-smi.jpg)

# 解决方案

如何解决这种问题呢？在 Nvidia 提出的分布式框架 Apex 里面，我们在源码里面找到了一个简单的解决方案：
[NVIDIA/apex](https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256)

```python?linenums
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()
	def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
```

我们能看到 Nvidia 是在读取每次数据返回给网络的时候，预读取下一次迭代需要的数据，那么对我们自己的训练代码只需要做下面的改造：

```python?linenums
training_data_loader = DataLoader(
    dataset=train_dataset,
    num_workers=opts.threads,
    batch_size=opts.batchSize,
    pin_memory=True,
    shuffle=True,
)
for iteration, batch in enumerate(training_data_loader, 1):
    # 训练代码

#-------------升级后---------

data, label = prefetcher.next()
iteration = 0
while data is not None:
    iteration += 1
    # 训练代码
    data, label = prefetcher.next()
```

这样子我们的 Dataloader 就像打了鸡血一样提高了效率很多，如下图：
![nvidia-smi2](./images/nvidia-smi2.jpg)

# 其他方案

## 预处理提速

> * 尽量减少每次读取数据时的预处理操作，可以考虑把一些固定的操作，例如 resize ，事先处理好保存下来，训练的时候直接拿来用
> * Linux上将预处理搬到GPU上加速：NVIDIA/DALI

## IO提速

> * 使用更快的图片处理库
> * &emsp;&emsp; opencv一般要比 PIL 要快
> * &emsp;&emsp; 对于jpeg读取，可以尝试 jpeg4py
> * &emsp;&emsp; 存bmp图（降低解码时间）
> * 小图拼起来存放（降低读取次数，对于大规模的小文件读取，建议转成单独的文件，可以选择的格式：TFRecord（Tensorflow）、recordIO（recordIO）、hdf5、 pth、n5、lmdb 等等 [Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch#data-loader)
> * &emsp;&emsp;[TFRecord](https://github.com/vahidk/tfrec)
> * &emsp;&emsp;借助lmdb数据库：
>   &emsp;&emsp;&emsp;&emsp;[Image2LMDB](https://github.com/Fangyh09/Image2LMDB)
>   &emsp;&emsp;&emsp;&emsp;[LMDB](https://blog.csdn.net/P_LarT/article/details/103208405)
>   &emsp;&emsp;&emsp;&emsp;[PySODToolBox](https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py)
> * 借助内存：直接载到内存里面，或者把把内存映射成磁盘
> * 借助固态：把读取速度慢的机械硬盘换成 NVME 固态

## 训练策略

> * 在训练中使用低精度（FP16甚至INT8、二值网络）表示取代原有精度（FP32）表示
> * [NVIDIA/Apex使用](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729)
> * torch.backends.cudnn.benchmark = True
> * Do numpy-like operations on the GPU wherever you can
> * Free up memory using del
> * Avoid unnecessary transfer of data from the GPU
> * Use **pinned memory**, and use non_blocking=False to parallelize data transfer and GPU number crunching

## 模型设计

来自ShuffleNetV2的结论：
`内存访问消耗时间，memory access cost 缩写为 MAC。`

> * 卷积层输入输出通道一致：卷积层的输入和输出特征通道数相等时MAC最小，此时模型速度最快
> * 减少卷积分组：过多的group操作会增大MAC，从而使模型速度变慢
> * 减少模型分支：模型中的分支数量越少，模型速度越快
> * 减少element-wise操作：element-wise操作所带来的时间消耗远比在FLOPs上的体现的数值要多，因此要尽可能减少element-wise操作（depthwise convolution也具有低FLOPs、高MAC的特点）

其他：

> * 降低复杂度：例如模型裁剪和剪枝，减少模型层数和参数规模
> * 改模型结构：例如模型蒸馏，通过知识蒸馏方法来获取小模型

本文来源：
- [如何给你PyTorch里的Dataloader打鸡血](https://zhuanlan.zhihu.com/p/66145913)
- [使用pytorch时，训练集数据太多达到上千万张，Dataloader加载很慢怎么办?](https://www.zhihu.com/question/356829360)
