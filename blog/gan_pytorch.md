# pytorch实现GAN



# 前言

在进入技术层面之前，为照顾新入门的开发者，先来介绍下什么是 GAN。

2014 年，Ian Goodfellow 和他在蒙特利尔大学的同事发表了一篇震撼学界的论文。没错，我说的就是《Generative Adversarial Nets》，这标志着生成对抗网络（GAN）的诞生，而这是通过对计算图和博弈论的创新性结合。他们的研究展示，给定充分的建模能力，两个博弈模型能够通过简单的反向传播（backpropagation）来协同训练。

这两个模型的角色定位十分鲜明。给定真实数据集 R，G 是生成器（generator），它的任务是生成能以假乱真的假数据；而 D 是判别器 （discriminator），它从真实数据集或者 G 那里获取数据， 然后做出判别真假的标记。Ian Goodfellow 的比喻是，G 就像一个赝品作坊，想要让做出来的东西尽可能接近真品，蒙混过关。而 D 就是文物鉴定专家，要能区分出真品和高仿（但在这个例子中，造假者 G 看不到原始数据，而只有 D 的鉴定结果——前者是在盲干）。
![gan](./images/gan.png)

理想情况下，D 和 G 都会随着不断训练，做得越来越好——直到 G 基本上成为了一个“赝品制造大师”，而 D 因无法正确区分两种数据分布输给 G。

实践中，Ian Goodfellow 展示的这项技术在本质上是：G 能够对原始数据集进行一种无监督学习，找到以更低维度的方式（lower-dimensional manner）来表示数据的某种方法。而无监督学习之所以重要，就好像 Yann LeCun 的那句话：“无监督学习是蛋糕的糕体”。这句话中的蛋糕，指的是无数学者、开发者苦苦追寻的“真正的 AI”。

# 核心思想

判断器的任务是尽力将假的判断为假的，将真的判断为真的；生成器的任务是使生成的越真越好。两者交替迭代训练。

# 核心代码

```python?linenums
    
real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1

fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0

 

# 1, D: 真的判断为真，假的判断为假

real_out = D(real_img)  # 将真实图片放入判别器中

d_loss_real = criterion(real_out, real_label)  # 真的判断为真

 

fake_img = G(z)  # 随机噪声放入生成网络中，生成一张假的图片

fake_out = D(fake_img)  # 判别器判断假的图片，

d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss

 

d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失

# 然后反向传播

 

# 2, G:生成的越真越好

fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片

fake_out = D(fake_img)  # 经过判别器得到的结果

g_loss = criterion(fake_out, real_label)  # 得到的假的图片与真实的图片的label的loss

# 然后反向传播
```

# 用 PyTorch 训练 GAN

```python?linenums
# coding=utf-8

import torch.autograd

import torch.nn as nn

from torch.autograd import Variable

from torchvision import transforms

from torchvision import datasets

from torchvision.utils import save_image

import os

 

# 创建文件夹

if not os.path.exists('./img'):

    os.mkdir('./img')

 

 

def to_img(x):

    out = 0.5 * (x + 1)

    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：

    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行

    return out

 

 

batch_size = 128

num_epoch = 100

z_dimension = 100

# 图像预处理

img_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std

])

 

# mnist dataset mnist数据集下载

mnist = datasets.MNIST(

    root='./data/', train=True, transform=img_transform, download=True

)

 

# data loader 数据载入

dataloader = torch.utils.data.DataLoader(

    dataset=mnist, batch_size=batch_size, shuffle=True

)

 

 

# 定义判别器  #####Discriminator######使用多层网络来作为判别器

# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，

# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。

class discriminator(nn.Module):

    def __init__(self):

        super(discriminator, self).__init__()

        self.dis = nn.Sequential(

            nn.Linear(784, 256),  # 输入特征数为784，输出为256

            nn.LeakyReLU(0.2),  # 进行非线性映射

            nn.Linear(256, 256),  # 进行一个线性映射

            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),

            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，

            # sigmoid可以班实数映射到【0,1】，作为概率值，

            # 多分类用softmax函数

        )

 

    def forward(self, x):

        x = self.dis(x)

        return x

 

 

# ###### 定义生成器 Generator #####

# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,

# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，

# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布

# 能够在-1～1之间。

class generator(nn.Module):

    def __init__(self):

        super(generator, self).__init__()

        self.gen = nn.Sequential(

            nn.Linear(100, 256),  # 用线性变换将输入映射到256维

            nn.ReLU(True),  # relu激活

            nn.Linear(256, 256),  # 线性变换

            nn.ReLU(True),  # relu激活

            nn.Linear(256, 784),  # 线性变换

            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布

        )

 

    def forward(self, x):

        x = self.gen(x)

        return x

 

 

# 创建对象

D = discriminator()

G = generator()

if torch.cuda.is_available():

    D = D.cuda()

    G = G.cuda()

 

 

# 首先需要定义loss的度量方式  （二分类的交叉熵）

# 其次定义 优化函数,优化函数的学习率为0.0003

criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)

g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

 

# ##########################进入训练##判别器的判断过程#####################

for epoch in range(num_epoch):  # 进行多个epoch的训练

    for i, (img, _) in enumerate(dataloader):

        num_img = img.size(0)

        # view()函数作用是将一个多行的Tensor,拼接成一行

        # 第一个参数是要拼接的tensor,第二个参数是-1

        # =============================训练判别器==================

        img = img.view(num_img, -1)  # 将图片展开为28*28=784

        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中

        real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1

        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0

 

        # ########判别器训练train#####################

        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假

        # 计算真实图片的损失

        real_out = D(real_img)  # 将真实图片放入判别器中

        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss

        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

        # 计算假的图片的损失

        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声

        fake_img = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离

        fake_out = D(fake_img)  # 判别器判断假的图片，

        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss

        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好

        # 损失函数和优化

        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失

        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0

        d_loss.backward()  # 将误差反向传播

        d_optimizer.step()  # 更新参数

 

        # ==================训练生成器============================

        # ###############################生成网络的训练###############################

        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，

        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，

        # 反向传播更新的参数是生成网络里面的参数，

        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的

        # 这样就达到了对抗的目的

        # 计算假的图片的损失

        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声

        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片

        output = D(fake_img)  # 经过判别器得到的结果

        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss

        # bp and optimize

        g_optimizer.zero_grad()  # 梯度归0

        g_loss.backward()  # 进行反向传播

        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

 

        # 打印中间的损失

        if (i + 1) % 100 == 0:

            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '

                  'D real: {:.6f},D fake: {:.6f}'.format(

                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),

                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值

            ))

        if epoch == 0:

            real_images = to_img(real_img.cpu().data)

            save_image(real_images, './img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)

    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

 

# 保存模型

torch.save(G.state_dict(), './generator.pth')

torch.save(D.state_dict(), './discriminator.pth')
```








本文来源：

> * [pytorch实现GAN](https://blog.csdn.net/jizhidexiaoming/article/details/96485095)

参考链接：

> * [用Pytorch实现WGAN](https://www.jianshu.com/p/ddfd7fba11d0)
> * [GAN入门实践（二）--Pytorch实现](https://www.codercto.com/a/31436.html)
