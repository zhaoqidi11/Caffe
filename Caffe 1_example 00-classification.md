因为目前只配置了CPU模式，所以自己把代码放到VS2015的python里面跑了。
```
import numpy as np
import matplotlib.pyplot as plt

import caffe

#设置默认显示参数
plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出

plt.show()

import sys

caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')

import caffe

caffe.set_mode_cpu()

model_def = caffe_root + '\\caffe\\models\\bvlc_reference_caffenet\\deploy.prototxt'
model_weights = caffe_root + '\\caffe\\models\\bvlc_reference_caffenet\\bvlc_reference_caffenet.caffemodel'

print model_def


net = caffe.Net(model_def,      # 定义模型结构
                model_weights,  # 包含了模型的训练权值
                caffe.TEST)     # 使用测试模式(不执行dropout),还有一个参数时caffe.TRAIN

 # 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + '\\caffe\\python\\caffe\\imagenet\\ilsvrc_2012_mean.npy')#np.load读取数组
print mu


mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
print 'mean-subtracted values:', zip('BGR', mu)
'''
# 对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})# caffe.io.Transformer 是用于预处理的类。事先填写进行预处理的要求。

transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
'''
```
