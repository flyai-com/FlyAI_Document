# 快速实现语音识别



# 介绍

本文旨在介绍如何简单的搭建一个语音识别系统，基于pyaudio利用Python编程从电脑端录制音频保存到指定文件夹+将录音上传服务器+录音进行识别并转为文本保

## 代码实现

```python?linenums
# -*- coding: utf-8 -*-

 

#pyaudio：利用pyaudio从电脑端录制音频保存到指定文件夹+将录音上传服务器+录音进行识别并转为文本保存

import wave

from pyaudio import PyAudio,paInt16

 

import urllib  #urllib2

import pycurl

import urllib.request as urllib2

import json 

 

framerate=8000   #采样率

NUM_SAMPLES=2000 #采样点

channels=1  #一个声道

sampwidth=2 #两个字节十六位

TIME=2      #条件变量，可以设置定义录音的时间

 

def save_wave_file(filename, data):   #save the date to the wav file

    wf = wave.open(filename, 'wb')  #二进制写入模式

    wf.setnchannels(channels)  

    wf.setsampwidth(sampwidth)  #两个字节16位

    wf.setframerate(framerate)  #帧速率

    wf.writeframes(b"".join(data))  #把数据加进去，就会存到硬盘上去wf.writeframes(b"".join(data)) 

    wf.close()

 

def my_record():

    pa=PyAudio()

    stream=pa.open(format=paInt16,channels=1,rate=framerate,input=True,frames_per_buffer=NUM_SAMPLES)

    my_buf=[]

    count=0  #

    while count < TIME*8: #循环2*20次

        string_audio_data=stream.read(NUM_SAMPLES) #每读完2000个采样加1

        my_buf.append(string_audio_data)

        count+=1

        print('当前正在录音(同时录制系统内部和麦克风的声音)……')

    save_wave_file('03.wav',my_buf) #文件保存

    stream.close()

    

def dump_res(buf):  #dump_res即dump_result,buf是curl从网上返回来的缓存

    print(buf)

    

    my_temp=json.loads(buf)

    my_list=my_temp['result']

    print(type(my_list))

    print(my_list[0])  #输出第一个

    print('dump_res函数调用成功！')

    

def get_token():  #获取token

    apikey='2KeNr6nK6ZmMKAbdlM5PUaSC'

    secretkey='QuDTqg1cMehfwvvyKmZyifAnCoGFiZ3g'

    auth_url='https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id='+apikey+'&client_secret='+secretkey;   #

    

    res=urllib2.urlopen(auth_url) #获取服务器响应,res=urllib2.urlopen(auth_url) 

    json_data=res.read()         #读取到json_data中

    print(json_data,type(json_data))

    return json.loads(json_data)['access_token']

 

def use_cloud(token):  #token类似一种访问权限等

    fp=wave.open(u'16k.wav','rb')             #打开wav文件

    nf=fp.getnframes()                     #获得文件的采样点数量

    print('sampwidth',fp.getsampwidth())

    print('framerate',fp.getframerate())

    print('channels',fp.getnchannels())

    f_len=nf*2                    #获取文件长度,文件长度计算，每个采样点2个字节

    audio_data=fp.readframes(nf)  #

    

    cuid="XXXXXXXXXX"   #硬件地址，my phone xiaomi MAC

    print(token)

    srv_url='http://vop.baidu.com/server_api'+'?cuid='+cuid+'&token='+token

    http_header=[

        'Content-Type:audio/pcm;rate=8000',  #音频,原先是pcm,可以改为wav

        'Content-length:%d:' % f_len

    ]

    

    c=pycurl.Curl()  #实例化curl

    c.setopt(pycurl.URL,str(srv_url))     #(网址)  

    

    c.setopt(c.HTTPHEADER, http_header)   #网址头部  

    c.setopt(c.POST, 1)                   #1表示调用post方法而不是get  

    c.setopt(c.CONNECTTIMEOUT,80)      #超时中断  

    c.setopt(c.TIMEOUT,80)             #下载超时  

    c.setopt(c.WRITEFUNCTION,dump_res) #返回数据，dump_res,进行回调  

    c.setopt(c.POSTFIELDS,audio_data)    #数据  

    c.setopt(c.POSTFIELDSIZE,f_len)      #文件大小

    c.perform()                           #提交， pycurl.perform()

    print('use_cloud函数over！')

 

if __name__ == "__main__": 

#     my_record()

    print('录音结束！')

    token = get_token() 

    use_cloud(token)

    print('over！')

```

本文来源：

> * [利用python进行语音识别](https://blog.csdn.net/qq_41185868/article/details/80496939)

