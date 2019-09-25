# 常见问题test



**Q：如何获得奖金？**

A：超过项目设置的最低分，根据公式计算，就可以获得奖金。

**Q：比赛使用什么框架？**

A：比赛支持常用的机器学习和深度学习框架，比如TensorFlow，PyTorch，Keras，Scikit-learn等。

**Q：怎么参加比赛，需不需要提交csv文件？**

A：FlyAI竞赛平台无需提交csv文件，在网页上点击报名，下载项目，使用你熟练的框架，修改`main.py`中的网络结构，和`processor.py`中的数据处理。使用FlyAI提供的命令提交，就可以参加比赛了。

**Q：比赛排行榜分数怎么得到的？**

A：参加项目竞赛必须实现 `model.py` 中的`predict_all`方法。系统通过该方法，调用模型得出评分。

**Q：平台机器什么配置？**

A：目前每个训练独占一块P40显卡，显存24G。

**Q：本地数据集在哪？**

A：运行 `flyai.exe test` or `./flyai test` 命令之后会下载100条数据到项目的data目录下，也可以本地使用ide运行 `main.py` 下载数据

**Q：FlyAI自带的Python环境在哪,会不会覆盖本地环境？**

A：FlyAI不会覆盖本地现有的Python环境。

* windows用户:

  C:Users{你计算机用户名}.flyaienvpython.exe

  C:Users{你计算机用户名}.flyaienvScriptspip.exe

* mac和linux用户:

  /Users/{你计算机用户名}/.flyai/env/bin/python3.6

  /Users/{你计算机用户名}/.flyai/env/bin/pip

**Q：FAI训练积分不够用怎么办？**

A：目前GPU免费使用，可以进入到：[我的积分](https://www.flyai.com/personal_score)，通过签到和分享等途径获得大量积分。

**Q：离线训练代码不符合规范问题**

A：`main.py`中必须使用`args.EPOCHS`和`args.BATCH`。

**Q：项目什么时候开始，持续多长时间？**

A：网站上能看见的项目就是已经开始的，项目会一直存在，随时可以提交。

**Q：排行榜和获奖问题**

A：目前可能获得奖金的项目是审核之后才能上榜的，每天会审核一次。通过审核之后，奖金才能发送到账户。

**Q：全量数据集怎么查看，数据集支不支持下载？**

A：暂时不支持查看和下载，如果需要，可以进入数据集来源链接查看。运行 `flyai.exe train -e=xx -b=xx` or `./flyai train -e=xx -b=xx` 命令之后会提交代码到服务器上使用全量数据集训练。

**Q：from flyai.dataset import Dataset 报错、No module name "flyai"**

A：先找到ide中使用的Python对应的pip的位置。

* windows用户：pip所在路径pip.exe install -i [https://pypi.flyai.com/simple](https://pypi.flyai.com/simple) flyai
* mac和linux用户：pip所在路径/pip install -i [https://pypi.flyai.com/simple](https://pypi.flyai.com/simple) flyai
* 其他 No module name "xxxx"问题 也可以参考上面 

