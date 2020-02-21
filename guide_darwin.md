1.进入代码编辑页下载当前代码

2.使用终端进入到项目的根目录下

> cd path\to\project

3.初始化环境登录

> 使用如下命令授权 flyai 脚本：
> chmod +x ./flyai

4.开启 FlyAI-Jupyter 代码调试环境

> 在终端执行命令 ./flyai ide 打开调试环境
> 操作过程有延迟，请耐心等待
> 运行 run main.py 命令即可在本地训练调试代码

5.提交训练到GPU

> 在FlyAI-Jupyter环境下运行  !flyai.exe train  将代码提交到云端GPU免费训练
> 返回sucess状态，代表提交离线训练成功
> 训练结束会以微信和邮件的形式发送结果通知

6.下载本地测试数据

> 首次成功执行本地调试命令后，将在本地代码包中自动生成"data"数据集文件夹


[更多参赛帮助请查看文档中心](http://doc.flyai.com/)
