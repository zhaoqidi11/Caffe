因为目前只配置了CPU模式，所以自己把代码放到VS2015的python里面跑了。
```Python
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
#输入的维度由四个input_dim字段构建。在默认情况下，使用的CaffeNet,net.blobs['data'].data.shape == (10, 3, 227, 227)，即从256*256的图像中提取出
#10个随机的227*227的crop（图像crop就是指从图像中移除不需要的信息，只保留需要的部分），进入网络。

transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
#transformer.set_transpose是转换图像维度。通常当一个图像库（image library），加载的数组的维数是(H*W*C)（其中H为高，W为宽，C为通道数），而Caffe是
#期望输入到其中的数据是C*H*W（这种表现方式），所以使用这个函数将数据进行转置，transformer.set_transpose('data',(2,0,1))使得第0维被第2维替换；第1
#维被第0维替换，第2维被第1维替换。

transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
#在理论上，我们应使用ILSVRC数据集的平均值，因为预训练的Caffenet/Googlenet/VGG在该图像上进行了训练。这与我们之前载入的ilsvrc_2012_mean.npy的文件相
#对应，如果为了效果更好,可以使用数组[104,117,123]，这是因为我们需要遵循在训练期间使用的标准化。而且，由自然图像组成的任何数据集的平均值应该接近
#[104,117,123]；当然，如果在不同于ILSVRC的数据集上从头开始训练网络，她需要使用该数据集的平均集。

transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
#caffe.io.load_image以标准化形式（0-1）加载数据，其中在示例中使用的模型是在正常图像值0-255上训练的。 提供参数255以告知transformer将值重新缩放回0-
#255范围。

transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
#对于该函数的理解类似transformer.set_transpose('data',(2,0,1))
'''
```
