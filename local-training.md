# 如何在本地调试代码和提交训练

### 如何下载代码包到本地？

**请在代码编辑页点击“下载代码“按钮，将自动下载当前代码压缩包**

![](.gitbook/assets/xia-zai-dai-ma.png)

### Win系统用户本地调试说明

#### 使用Windows客户端

* 解压下载的样例代码包，并打开“flyai.exe“程序
* 首次打开需要使用FlyAI账号登录
* 登录成功后可以下载调试数据集，调试数据下载完毕将会在文件夹内生成以“data“命名的文件夹
* 在GUI操作界面使用本地调试命令进行代码调试

查看Windows客户端使用教程视频：

[https://dataset.flyai.com/flyai\_course-localdebugging.mp4](https://dataset.flyai.com/flyai_course-localdebugging.mp4)

#### 使用Windows 终端

* 下载并解压代码包
* 打开运行，输入 cmd ,打开终端

```text
Win + R 输入 cmd 打开终端
```

* 使用终端进入到项目的根目录下,如：

```text
cd 代码包根目录
# 快捷方式：在终端输入"cd"后空格，直接将代码包文件拖拽到终端窗口中
```

* 执行（test）本地调试命令，自动配置本地环境依赖（首次登录需使用微信扫码登录FlyAI账号）

```text
flyai.exe test
```

* 提交本地代码文件到云端GPU训练

```text
flyai.exe train
```

### Mac/Linux系统用户本地调试说明

* 下载并解压代码包
* 使用终端进入到项目的根目录下,如：

```text
cd 代码包根目录
# 快捷方式：在终端输入"cd"后空格，直接将代码包文件拖拽到终端窗口中
```

* 授权flyai

```text
chmod +x ./flyai
```

* 执行（test）本地调试命令，自动配置本地环境依赖（首次登录需使用微信扫码登录FlyAI账号）

```text
./flyai test
```

* 提交本地代码文件到云端GPU训练

```text
./flyai train
```



