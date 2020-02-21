1.进入代码编辑页下载当前代码

2.本地解压缩代码包文件，双击执行 flyai.exe 程序

> 第一次使用需要使用微信扫码登录
> 杀毒软件可能会误报，点击信任该程序即可

3.开启 FlyAI-Jupyter 代码调试环境

> 运行flyai.exe程序，点击"使用jupyter调试"按钮自动打开jupyter lab 操作界面

> 运行 run main.py 命令即可在本地训练调试代码
>
> 如果出现 No Model Name "xxx"错误，需在 requirements.txt 填写项目依赖

4.下载本地测试数据

> 运行flyai.exe程序，点击"下载数据"按钮，程序会下载100条调试数据

5.提交训练到GPU

> 运行flyai.exe程序，点击"提交GPU训练"按钮，代码将自动提交到云端GPU进行训练
>
> 返回sucess状态，代表提交离线训练成功，训练结束会以微信和邮件的形式发送结果通知
>
> 项目中有新的Python包引用，必须在 requirements.txt 文件中指定包名，不填写版本号将默认安装最新版
>




[更多参赛帮助请查看文档中心](http://doc.flyai.com/)
