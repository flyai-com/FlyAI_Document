1.进入代码编辑页下载当前代码

2.使用终端进入到项目的根目录下

> cd path\to\project

3.初始化环境登录

> 使用如下命令授权 flyai 脚本：
> chmod +x ./flyai

4.开启 Jupyter 代码调试环境

> 在终端执行命令 ./flyai ide 打开调试环境（第一次使用需要使用微信扫码登录）
> 操作过程有延迟，请耐心等待
>
> 运行 run main.py 命令即可在本地训练调试代码
>
> 如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

或使用终端

>执行下列命令本地安装环境并调试
>
>./flyai test 注意:如果出现 permission denied 错误，需使用sudo运行
>
>如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖
>
>执行test命令，会自动下载100条测试数据到项目下
>
>如果使用本地IDE开发，可以自行安装 requirements.txt 中的依赖，运行 main.py 即可

5.提交训练到GPU

> 在FlyAI-Jupyter环境下运行  !./flyai train  将代码提交到云端GPU免费训练
> 返回sucess状态，代表提交离线训练成功，训练结束会以微信和邮件的形式发送结果通知
> 
>或者在终端下使用 ./flyai train -b=32 -e=10 
> 
>项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，不填写版本号将默认安装最新版

6.下载本地测试数据

> 首次成功执行本地调试命令后，将在本地代码包中自动生成"data"数据集文件夹

7.使用自己的Python环境

> ./flyai path=xxx 可以设置自己的Python路径
>
> ./flyai path=flyai 恢复系统默认Pyton路径



[更多参赛帮助请查看文档中心](http://doc.flyai.com/)

