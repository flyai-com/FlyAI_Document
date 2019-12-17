# 二代身份证号识别


# 前言

最近在研究OCR识别相关的东西，最终目标是能识别身份证上的所有中文汉字+数字，不过本文先设定一个小目标，先识别定长为18的身份证号，当然本文的思路也是可以复用来识别定长的验证码识别的。
本文实现思路主要来源于Xlvector的博客，采用基于CNN实现端到端的OCR，下面引用博文介绍目前基于深度学习的两种OCR识别方法：

> 1. 把OCR的问题当做一个多标签学习的问题。4个数字组成的验证码就相当于有4个标签的图片识别问题（这里的标签还是有序的），用CNN来解决。
> 2. 把OCR的问题当做一个语音识别的问题，语音识别是把连续的音频转化为文本，验证码识别就是把连续的图片转化为文本，用CNN+LSTM+CTC来解决。

这里方法1主要用来解决固定长度标签的图片识别问题，而方法2主要用来解决不定长度标签的图片识别问题，本文实现方法1识别固定18个数字字符的身份证号。

# 环境依赖

1. 本文基于tensorflow框架实现,依赖于tensorflow环境，建议使用anaconda进行python包管理及环境管理
2. 本文使用freetype-py 进行训练集图片的实时生成，同时后续也可扩展为能生成中文字符图片的训练集，建议使用pip安装

```python
  pip install freetype-py
```

同时本文还依赖于numpy和opencv等常用库。

# 训练数据集生成

首先先完成训练数据集图片的生成，主要依赖于freetype-py库生成数字/中文的图片。其中要注意的一点是就是生成图片的大小，本文经过多次尝试后，生成的图片是32 x 256大小的，如果图片太大，则可能导致训练不收敛。
gen_image()方法返回
image_data：图片像素数据 (32,256)
label： 图片标签 18位数字字符 477081933151463759
vec : 图片标签转成向量表示 (180,) 代表每个数字所处的列，总长度 18 * 10

```python?linenums
#!/usr/bin/env python2

# -*- coding: utf-8 -*-

"""

身份证文字+数字生成类



@author: pengyuanjie

"""

import numpy as np

import freetype

import copy

import random

import cv2

 

class put_chinese_text(object):

    def __init__(self, ttf):

        self._face = freetype.Face(ttf)

 

    def draw_text(self, image, pos, text, text_size, text_color):

        '''

        draw chinese(or not) text with ttf

        :param image:     image(numpy.ndarray) to draw text

        :param pos:       where to draw text

        :param text:      the context, for chinese should be unicode type

        :param text_size: text size

        :param text_color:text color

        :return:          image

        '''

        self._face.set_char_size(text_size * 64)

        metrics = self._face.size

        ascender = metrics.ascender/64.0

 

        #descender = metrics.descender/64.0

        #height = metrics.height/64.0

        #linegap = height - ascender + descender

        ypos = int(ascender)

 

        if not isinstance(text, unicode):

            text = text.decode('utf-8')

        img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)

        return img

 

    def draw_string(self, img, x_pos, y_pos, text, color):

        '''

        draw string

        :param x_pos: text x-postion on img

        :param y_pos: text y-postion on img

        :param text:  text (unicode)

        :param color: text color

        :return:      image

        '''

        prev_char = 0

        pen = freetype.Vector()

        pen.x = x_pos << 6   # div 64

        pen.y = y_pos << 6

 

        hscale = 1.0

        matrix = freetype.Matrix(int(hscale)*0x10000L, int(0.2*0x10000L),\

                                 int(0.0*0x10000L), int(1.1*0x10000L))

        cur_pen = freetype.Vector()

        pen_translate = freetype.Vector()

 

        image = copy.deepcopy(img)

        for cur_char in text:

            self._face.set_transform(matrix, pen_translate)

 

            self._face.load_char(cur_char)

            kerning = self._face.get_kerning(prev_char, cur_char)

            pen.x += kerning.x

            slot = self._face.glyph

            bitmap = slot.bitmap

 

            cur_pen.x = pen.x

            cur_pen.y = pen.y - slot.bitmap_top * 64

            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

 

            pen.x += slot.advance.x

            prev_char = cur_char

 

        return image

 

    def draw_ft_bitmap(self, img, bitmap, pen, color):

        '''

        draw each char

        :param bitmap: bitmap

        :param pen:    pen

        :param color:  pen color e.g.(0,0,255) - red

        :return:       image

        '''

        x_pos = pen.x >> 6

        y_pos = pen.y >> 6

        cols = bitmap.width

        rows = bitmap.rows

 

        glyph_pixels = bitmap.buffer

 

        for row in range(rows):

            for col in range(cols):

                if glyph_pixels[row*cols + col] != 0:

                    img[y_pos + row][x_pos + col][0] = color[0]

                    img[y_pos + row][x_pos + col][1] = color[1]

                    img[y_pos + row][x_pos + col][2] = color[2]

 

 

class gen_id_card(object):

    def __init__(self):

       #self.words = open('AllWords.txt', 'r').read().split(' ')

       self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

       self.char_set = self.number

       #self.char_set = self.words + self.number

       self.len = len(self.char_set)

       

       self.max_size = 18

       self.ft = put_chinese_text('fonts/OCR-B.ttf')

       

    #随机生成字串，长度固定

    #返回text,及对应的向量

    def random_text(self):

        text = ''

        vecs = np.zeros((self.max_size * self.len))

        #size = random.randint(1, self.max_size)

        size = self.max_size

        for i in range(size):

            c = random.choice(self.char_set)

            vec = self.char2vec(c)

            text = text + c

            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)

        return text,vecs

    

    #根据生成的text，生成image,返回标签和图片元素数据

    def gen_image(self):

        text,vec = self.random_text()

        img = np.zeros([32,256,3])

        color_ = (255,255,255) # Write

        pos = (0, 0)

        text_size = 21

        image = self.ft.draw_text(img, pos, text, text_size, color_)

        #仅返回单通道值，颜色对于汉字识别没有什么意义

        return image[:,:,2],text,vec

 

    #单字转向量

    def char2vec(self, c):

        vec = np.zeros((self.len))

        for j in range(self.len):

            if self.char_set[j] == c:

                vec[j] = 1

        return vec

        

    #向量转文本

    def vec2text(self, vecs):

        text = ''

        v_len = len(vecs)

        for i in range(v_len):

            if(vecs[i] == 1):

                text = text + self.char_set[i % self.len]

        return text

 

if __name__ == '__main__':

    genObj = gen_id_card()

    image_data,label,vec = genObj.gen_image()

    cv2.imshow('image', image_data)

    cv2.waitKey(0)

```

# 构建网络，开始训练

首先定义生成一个batch的方法：

```python?linenums
# 生成一个训练batch

def get_next_batch(batch_size=128):

    obj = gen_id_card()

    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])

    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

 

 

    for i in range(batch_size):

        image, text, vec = obj.gen_image()

        batch_x[i,:] = image.reshape((IMAGE_HEIGHT*IMAGE_WIDTH))

        batch_y[i,:] = vec

    return batch_x, batch_y

```

使用Batch Normalization

```python?linenums
#Batch Normalization? 有空再理解,tflearn or slim都有封装

## http://stackoverflow.com/a/34634291/2267819

def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):

    with tf.variable_scope(scope):

        #beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)

        #gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)

 

        def mean_var_with_update():

            ema_apply_op = ema.apply([batch_mean, batch_var])

            with tf.control_dependencies([ema_apply_op]):

                return tf.identity(batch_mean), tf.identity(batch_var)

 

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    return normed

```

定义4层CNN和一层全连接层，卷积核分别是2层5x5、2层3x3，每层均使用tf.nn.relu非线性化,并使用max_pool，网络结构读者可自行调参优化。

```python?linenums
# 定义CNN

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):

    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

 

    # 4 conv layer

    w_c1 = tf.Variable(w_alpha*tf.random_normal([5, 5, 1, 32]))

    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))

    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)

    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=0.02), train_phase, scope='bn_1')

    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv1 = tf.nn.dropout(conv1, keep_prob)

 

    w_c2 = tf.Variable(w_alpha*tf.random_normal([5, 5, 32, 64]))

    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))

    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)

    conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_2')

    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.dropout(conv2, keep_prob)

 

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))

    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))

    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)

    conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_3')

    conv3 = tf.nn.relu(conv3)

    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.dropout(conv3, keep_prob)

 

    w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))

    b_c4 = tf.Variable(b_alpha*tf.random_normal([64]))

    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)

    conv4 = batch_norm(conv4, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_4')

    conv4 = tf.nn.relu(conv4)

    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv4 = tf.nn.dropout(conv4, keep_prob)

     

    # Fully connected layer

    w_d = tf.Variable(w_alpha*tf.random_normal([2*16*64, 1024]))

    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))

    dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])

    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    dense = tf.nn.dropout(dense, keep_prob)

 

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))

    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))

    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out

```

最后执行训练，使用sigmoid分类，每100次计算一次准确率，如果准确率超过80%，则保存模型并结束训练。

```python?linenums
# 训练

def train_crack_captcha_cnn():

    output = crack_captcha_cnn()

    # loss

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))

    # 最后一层用来分类的softmax和sigmoid有什么不同？

    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰

    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

 

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)

    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    correct_pred = tf.equal(max_idx_p, max_idx_l)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

 

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

 

        step = 0

        while True:

            batch_x, batch_y = get_next_batch(64)

            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75, train_phase:True})

            print(step, loss_)

            

            # 每100 step计算一次准确率

            if step % 100 == 0 and step != 0:

                batch_x_test, batch_y_test = get_next_batch(100)

                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., train_phase:False})

                print  "第%s步，训练准确率为：%s" % (step, acc)

                # 如果准确率大80%,保存模型,完成训练

                if acc > 0.8:

                    saver.save(sess, "crack_capcha.model", global_step=step)

                    break

            step += 1

```

# 后记

最后所有代码和字体资源文件托管在我的[Github](https://github.com/jimmyleaf/ocr_tensorflow_cnn)下。
笔者在一开始训练的时候图片大小是64 x 512的，训练的时候发现训练速度很慢，而且训练的loss不收敛一直保持在0.33左右，缩小图片为32 x 256后解决，不知道为啥，猜测要么是网络层级不够，或者特征层数不够吧。





本文来源：

> * [tensorflow 实现端到端的OCR：二代身份证号识别](https://blog.csdn.net/javastart/article/details/86482562)
