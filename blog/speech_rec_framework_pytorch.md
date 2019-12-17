# PyTorch语音识别框架



# 介绍

patter，一个PyTorch中的语音到文本框架，初始支持DeepSpeech2架构（及其变体）。

# 特征

>  * 基于文件的语料库定义配置，模型体系结构和可重复性的培训配置
>  * DeepSpeech模型具有高度可配置性
>  * 各种RNN类型（RNN，LSTM，GRU）和大小（层/隐藏单元）
>  * 各种激活功能（Clipped ReLU，Swish）
>  * 具有Lookahead（用于流式传输）或双向RNN的仅向前RNN
>  * 可配置的CNN前端
>  * 可选的batchnorm
>  * 可选的RNN重量噪音
>  * 具有KenLM支持的波束解码器
>  * 数据集扩充，支持：
>  * 速度扰动
>  * 获得扰动
>  * 移动（及时）扰动
>  * 噪声添加（随机SNR）
>  * 脉冲响应扰动
>  * Tensorboard集成
>  * 基于gRPC的模型服务器

# 安装

需要手动安装两个依赖项：

> * SeanNaren / warp-ctc和包含在回购中的pytorch绑定
> * parlance / ctcdecode CTC波束解码器支持语言模型

一旦安装了这些依赖项，就可以通过简单运行来安装模式python setup.py install。出于调试和开发目的，可以安装模式python setup.py develop。

# 数据集定义

使用带有换行符分隔的json对象的json-lines文件定义模式的数据集。每个链接都包含一个json对象，它定义了一个话语的音频路径，转录路径和持续时间（以秒为单位）。

```python?linenums
{"audio_filepath": "/path/to/utterance1.wav", "text_filepath": "/path/to/utterance1.txt", "duration": 23.147}
{"audio_filepath": "/path/to/utterance2.wav", "text_filepath": "/path/to/utterance2.txt", "duration": 18.251}
```

# 训练

Patter包括一个顶级训练器脚本，该脚本调用底层库方法进行训练。要使用内置命令行培训师，必须定义三个文件：语料库配置，模型配置和培训配置。以下提供各自的实例。

# 语料库配置

```python?linenums
# Filter the audio configured in the `datasets` below to be within min and max duration. Remove min or max (or both) to
# do no filtering
min_duration = 1.0
max_duration = 17.0
# Link to manifest files (as described above) of the training and validation sets. A future release will allow multiple
# files to be specified for merging corpora on the fly. If `augment` is true, each audio will be passed through the 
# augmentation pipeline specified below. Valid names for the datasets are in the set ["train", "val"]
[[dataset]]
name = "train"
manifest = "/path/to/corpora/train.json"
augment = true
[[dataset]]
name = "val"
manifest = "/path/to/corpora/val.json"
augment = false
# Optional augmentation pipeline. If specified, audio from a dataset with the augment flag set to true will be passed
# through each augmentation, in order. Each augmentation must minimally specify the type and a probability. The 
# probability indicates that the augmentation will run on a given audio file with that probability
# The noise augmentation mixes audio from a dataset of noise files with a random SNR drawn from within the range specified.
[[augmentation]]
type = "noise"
prob = 0.0
[augmentation.config]
manifest = "/path/to/noise_manifest.json"
min_snr_db = 3
max_snr_db = 35
# The impulse augmentation applies a random impulse response drawn from the manifest to the audio 
[[augmentation]]
type = "impulse"
prob = 0.0
[augmentation.config]
manifest = "/path/to/impulse_manifest.json"
# The speed augmentation applies a random speed perturbation without altering pitch
[[augmentation]]
type = "speed"
prob = 1.0
[augmentation.config]
min_speed_rate = 0.95
max_speed_rate = 1.05
# The shift augmentation simply adds a random amount of silence to the audio or removes some of the initial audio
[[augmentation]]
type = "shift"
prob = 1.0
[augmentation.config]
min_shift_ms = -5
max_shift_ms = 5
# The gain augmentation modifies the gain of the audio by a fixed amount randomly chosen within the specified range
[[augmentation]]
type = "gain"
prob = 1.0
[augmentation.config]
min_gain_dbfs = -10
max_gain_dbfs = 10
```

# 型号配置

此时，patter仅支持DeepSpeech 2和DeepSpeech 3（与DS2 w / o BatchNorm + Weight Noise）架构相同的变体。未来版本中可能包含未来的模型体系结构，包括新颖的体系结构。要配置体系结构和超参数，请将模型定义为配置TOML。见例子：

```python?linenums
# model class - only DeepSpeechOptim currently
model = "DeepSpeechOptim"
# define input features/windowing. Currently only STFT is supported, but window is configurable.
[input]
type = "stft"
normalize = true
sample_rate = 16000
window_size = 0.02
window_stride = 0.01
window = "hamming"
# Define layers of [2d CNN -> Activation -> Optional BatchNorm] as a frontend
[[cnn]]
filters = 32
kernel = [41, 11]
stride = [2, 2]
padding = [0, 10]
batch_norm = true
activation = "hardtanh"
activation_params = [0, 20]
[[cnn]]
filters = 32
kernel = [21, 11]
stride = [2, 1]
padding = [0, 2]
batch_norm = true
activation = "hardtanh"
activation_params = [0, 20]
# Configure the RNN. Currently LSTM, GRU, and RNN are supported. QRNN will be added for forward-only models in a future release
[rnn]
type = "lstm"
bidirectional = true
size = 512
layers = 4
batch_norm = true
# DS3 suggests using weight noise instead of batch norm, only set when rnn batch_norm = false
#[rnn.noise]
#mean=0.0
#std=0.001
# only used/necessary when rnn bidirectional = false
#[context]
#context = 20
#activation = "swish"
# Set of labels for model to predict. Specifying a label for the CTC 'blank' symbol is not required and handled automatically
[labels]
labels = [
 "'", "A", "B", "C", "D", "E", "F", "G", "H",
 "I", "J", "K", "L", "M", "N", "O", "P", "Q",
 "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ",
]
```

# 测试

提供模式测试脚本用于对训练模型进行评估。它将测试配置和训练模型作为参数。

```python?linenums
cuda = true
batch_size = 10
num_workers = 4
[[dataset]]
name = "test"
manifest = "/path/to/manifests/test.jl"
augment = false
[decoder]
algorithm = "greedy" # or "beam"
workers = 4
# If `beam` is specified as the decoder type, the below is used to initialize the beam decoder
[decoder.beam]
beam_width = 30
cutoff_top_n = 40
cutoff_prob = 1.0
# If "beam" is specified and you want to use a language model, configure the ARPA or KenLM format LM and alpha/beta weights
[decoder.beam.lm]
lm_path = "/path/to/language/model.arpa"
alpha = 2.15
beta = 0.35
```

更多使用方法可以查看官方文档
开源地址：[patter](https://blog.csdn.net/weixin_42137700/article/details/101148113)

本文来源：

> * [PyTorch语音识别框架，将语音转成文本格式](https://blog.csdn.net/weixin_42137700/article/details/101148113)

