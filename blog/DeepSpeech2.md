# 使用DeepSpeech2进行语音识别



# 介绍

本文主要介绍使用DeepSpeech2实现 Baidu Warp-CTC ，创建基于deepSpeech2架构的网络，并使用CTC激活函数进行训练。
DeepSpeech2具有以下特性：

> * 训练DeepSpeech，可配置递归的类型和体系结构，支持多gpu支持
> * 使用kenlm的语言模型支持（现在的WIP，目前还没有建立LM的指令）
> * 多个数据集下载器，支持AN4 TED voxforge和Librispeech，数据集可以合并，支持自定义数据集
> * 在线训练噪声注入提高噪声鲁棒性
> * 音频增强以提高噪声稳健性
> * 训练过程中出现故障或停止时的重新启动功能
> * 可视化训练图的Visdom/Tensorboard支持

# 安装

Git URL: 

```bash
git://www.github.com/SeanNaren/deepspeech.pytorch.git
```

Git Clone代码到本地: 

```bash
git clone http://www.github.com/SeanNaren/deepspeech.pytorch
```

Subversion代码到本地: 

```bash
$ svn co --depth empty http://www.github.com/SeanNaren/deepspeech.pytorch
Checked out revision 1.
$ cd repo
$ svn up trunk
```

此外，需要安装几个库以便进行训练。我假设所有的东西都安装在Ubuntu上的Anaconda中。
如果你还没有安装[pytorch](https://github.com/pytorch/pytorch#installation)，请安装。
为Warp-CTC绑定安装这个fork ：

```bash
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding
python setup.py install
```

安装pytorch audio：

```bash
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
pip install cffi
python setup.py install
```

如果你希望支持使用语言模型进行beam搜索解码，请安装ctcdecode：

```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

最后克隆这个repo，并且在repo中运行它：

```bash
pip install -r requirements.txt
```

# Docker

没有正式的Dockerhub镜像，但是提供了一个Dockerfile来构建你自己的系统。

```bash?linenums
sudo nvidia-docker build -t deepspeech2.docker .
sudo nvidia-docker run -ti -v `pwd`/data:/workspace/data -p 8888:8888 deepspeech2.docker # Opens a Jupyter notebook, mounting the /data drive in the container
```

如果你更喜欢bash ：

```bash
nvidia-docker run -ti -v `pwd`/data:/workspace/data --entrypoint=/bin/bash deepspeech2.docker # Opens a bash terminal, mounting the /data drive in the container

```

# 数据集

目前支持AN4 Tedium VoxForge和Librispeech，脚本会设置数据集，并且创建dataloading中使用的清单文件。

## AN4

在repo的根文件夹中下载，并且设置运行下面的an4数据集：

```bash?linenums
cd data; python an4.py

```

## TEDLIUM

你可以选择手动下载原始数据集文件，也可以通过脚本(会缓存它)下载，在这里 找到文件。
在repo的根文件夹中下载，并且设置TEDLIUM_V2数据集运行命令：

```bash?linenums
cd data; python ted.py # Optionally if you have downloaded the raw dataset file, pass --tar_path /path/to/TEDLIUM_release2.tar.gz

```

## Voxforge

要下载和设置Voxforge数据集，请在repo的根文件夹中运行以下命令：

```bash?linenums
cd data; python voxforge.py

```

请注意，此数据集不带有验证数据集或测试数据集。

## LibriSpeech

下载和设置LibriSpeech数据集，在repo的根目录中运行以下命令：

```bash
cd data; python librispeech.py

```

你可以选择手动下载原始数据集文件，也可以通过脚本(也会缓存它们)下载，为此，你必须创建下面的文件夹结构，并从这里下载相应的tar文件 。

```bash?linenums
cd data/
mkdir LibriSpeech/ # This can be anything as long as you specify the directory path as --target-dir when running the librispeech.py script
mkdir LibriSpeech/val/
mkdir LibriSpeech/test/
mkdir LibriSpeech/train/

```

现在将tar.gz文件放入正确的文件夹中，它们现在将用于librispeech的数据预处理，并在格式化数据集后删除。
如果你不想添加所有文件，可以选择指定所需的librispeech文件，像下面这样：

```bash?linenums
cd data/
python librispeech.py --files-to-use"train-clean-100.tar.gz, train-clean-360.tar.gz,train-other-500.tar.gz, dev-clean.tar.gz,dev-other.tar.gz, test-clean.tar.gz,test-other.tar.gz"

```

## 自定义数据集

要创建自定义数据集，必须创建一个包含训练数据位置的CSV文件，必须是以下格式：

```bash?linenums
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...

```

## 合并多个清单文件

要创建更大的清单文件（一次训练/测试多个数据集），我们可以将包含所有要合并的清单的目录中的清单文件合并在一起，如下所示。你还可以从新清单中删除短剪辑和长剪辑。

```bash?linenums
cd data/
python merge_manifests.py --output-path merged_manifest.csv --merge-dir all-manifests/ --min-duration 1 --max-duration 15 # durations in seconds

```

# 训练

```bash?linenums
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv

```

使用python train.py --help获取更多参数和选项。
也有[visdom](https://github.com/facebookresearch/visdom)支持可视化训练，启动服务器后，使用：

```bash?linenums
python train.py --visdom

```

也有[tensorboard](https://github.com/lanpa/tensorboardX)支持可视化训练，按照说明设置，使用：

```bash?linenums
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory

```

对于两个可视化工具，你可以在训练时通过更改--id参数将自己的名称添加到运行中。

## 多gpu训练

我们支持通过分布式并行(查看这里和这里，以便了解为什么不使用DataParallel )进行多gpu训练。
要使用多gpu ：

```bash?linenums
python -m multiproc python train.py --visdom --cuda # Add your parameters as normal, multiproc will scale to all GPUs automatically

```

multiproc将打开除主进程以外的所有进程的日志。
如果Infiniband不可用，我们建议使用TCP默认的gloo后端，NCCL2也可以作为后端使用，这里有更多信息。

## 噪声增强和噪声注入

有两种不同类型的噪声支持，噪声增强和噪声注入。

> * 噪声增强：加载音频时对速度和增益做一些小更改以增强健壮性，使用时，使用--augment标志进行训练。
> * 噪声注入：在训练数据中动态添加噪声以增强鲁棒性，要使用它，得首先使用采样的噪声文件填充目录。数据加载器将从这个目录中随机选取样本。

要启用噪声注入，请使用--noise-dir/path/to/noise/dir/指定噪音文件的位置，有一些噪声参数，如--noise_prob确定噪声的概率，--noise-min，--noise-max参数确定训练中的最小噪声和最大噪声。
包括了一个脚本，用于将噪声注入音频文件，以听到不同的噪音级别和文件的声音，适用于安排噪音数据集。

```bash?linenums
python noise_inject.py --input-path /path/to/input.wav --noise-path /path/to/noise.wav --output-path /path/to/input_injected.wav --noise-level 0.5 # higher levels means more noise

```

## Checkpoints( 检查点)

训练支持保存模型的检查点，以便在发生错误或早期终止时继续训练，要启用epoch 检查点，请执行以下操作：

```bash?linenums
python train.py --checkpoint

```

要启用每个N批次的检查点，以及epoch保存，请执行以下操作：

```bash?linenums
python train.py --checkpoint --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.

```

请注意，要使batch检查点系统正常工作，在从原始训练运行中加载检查点模型时，无法更改batch size。
要从已保存的已保存的模型继续，请执行以下操作：

```bash?linenums
python train.py --continue-from models/deepspeech_checkpoint_epoch_N_iter_N.pth.tar

```

这将从相同的训练状态继续，并且重新创建visdom图继续（如果启用）。
如果你想从以前的模型检查点开始，但是不继续训练，请添加--finetune标志，从--continue-from权值重新开始训练。

## 选择batch size

脚本可以用来测试是否可以在硬件上进行训练，以及可以使用的model/batch size的限制，使用：

```shell?linenums
python benchmark.py --batch-size 32

```

使用标志--help查看可以与脚本一起使用的其他参数。

## 模型详细信息

保存的模型包含它训练进程的元数据，要查看元数据，请执行以下命令：

```shell?linenums
python model.py --model-path models/deepspeech.pth.tar

```

另外需要注意的是，在模型上没有最终的SoftMax层，因为在训练时Warp CTC会在内部执行SoftMax，如果在模型的顶部构建了任何东西，这也必须在复杂解的码器中实现，因此请考虑清楚！

# Testing/Inference

要在测试集(必须与训练集的格式相同)上评估经过训练的模型，请执行以下操作：

```shell?linenums
python test.py --model-path models/deepspeech.pth --test-manifest /path/to/test_manifest.csv --cuda

```

提供了一个输出的示例脚本：

```shell?linenums
python transcribe.py --model-path models/deepspeech.pth --audio-path /path/to/audio.wav

```

# 备用解码器

默认情况下，test.py和transcribe.py使用GreedyDecoder，它在每个时间点选择最高的可能输出标签，然后过滤重复和空白符号，给出最终输出。
可以根据安装部分的说明，选择使用beam搜索解码器来安装ctcdecode库，test和transcribe脚本有一个--decoder参数，要使用beam解码器，请添加--decoder beam ，beam解码器支持额外的解码参数：

> * beam_width每个时间步长要考虑多少个beam
> * lm_path可选的，用于解码二进制KenLM语言模型
> * 语言模型的alpha权重
> * 单词的beta加成权重

# 时间偏移

使用--offsets标志在使用transcribe.py脚本时获取转录中每个字符的位置信息，偏移量基于输出张量的大小，你需要将它转换为所需的格式，例如，基于默认参数，你可以将偏移值乘以标量（以秒为单位的文件持续时间/输出大小）以获得以秒为单位的偏移量。

# 致谢

感谢[Egor](https://github.com/EgorLakomkin)和[ryan](https://github.com/ryanleary)对他们的贡献！


本文来源：

> * [deepspeech.pytorch, 使用DeepSpeech2进行语音识别](https://www.helplib.com/GitHub/article_151182)
