### 1 准备工作
首先是准备工作，在cygwin里运行两个sh文件<br />
>
首先是<font color="#660000">根目录下的data\mnist\get_mnist.sh</font><br /> 
>(如果出现'\lr'命令不识别问题的话，用dos2unix转换该文件)<br />
一般网速没问题的话，很快就可以下载好了<br />
接下来要复杂一点<br />
>
修改caffe根目录下的examples\mnist\create_mnist.sh文件<br />
>
将BUILD的目录更改为<br />

```
BUILD=scripts/build/examples/mnist/Release
```
至于为什么BUILD目录会在这个目录下我也不太清楚，只是编译之后就这样了<br />
更改后，在cygwin中切换到**根目录/scripts/build/examples/mnist/Release**下，<br />
我们需要将convert_mnist_data.exe文件转换成.bin文件<br />
（该方法来自https://groups.google.com/forum/#!topic/caffe-users/sXT3TuGt8Nc
中的Steven Clark先生）<br />
输入命令<br />
```
ln -s convert_mnist_data convert_mnist_data.bin
```
完成后
切换到caffe的根目录下(一般是有examples，scripts等目录的那个目录）<br />
输入命令<br />
```
bash C:\\caffe\\caffe\\examples\\mnist\\create_mnist.sh
```
即可完成<br />
>

### 2 实现Lenet
接下来我们实现Lenet的变体，经典的1989年的卷积架构<br />

我们需要两个外部文件来辅助：<br />
①net prototxt，定义了网络的结构，以及train和test数据<br />
②solver prototxt,定义了学习的参数（learning parameters）<br />

我们要编写的这个网络从预生成的lmdb，也可以使用MemoryDataLayer直接从ndarrays中读取数据<br />

```Python

from pylab import *

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')
import caffe

caffe.set_mode_cpu()


import os
os.chdir(caffe_root)
## Download data，原本是要在py脚本中执行sh文件的，为了方便直接在cygwin中运行该sh文件
#!data/mnist/get_mnist.sh
## Prepare data

#import subprocess
#os.system('examples/mnist/create_mnist.sh')

# back to examples
os.chdir('examples')

from caffe import layers as L, params as P
#layers包含了caffe内置的层，params包含了各种枚举值


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    #我们的Lenet：一系列线性和简单的非线性变换
    
    #关于NetSpec的介绍：https://blog.csdn.net/u012762410/article/details/78899212
    #NetSpec 是包含Tops（可以直接赋值作为属性）的集合。
    #调用 NetSpec.to_proto 创建包含所有层(layers)的网络参数，这些层(layers)需要被赋值，并使用被赋值的名字。
    
    #n=caffe.NetSpec() 是获取Caffe的一个Net，我们只需不断的填充这个n，最后面把n输出到文件就会使我们在Caffe学习里面看到的Net的protobuf的定义。
    n = caffe.NetSpec()
    #部分详解来自http://yanjoy.win/2017/01/06/pycaffe-interface1/
    #部分详解来自https://www.cnblogs.com/JZ-Ser/articles/7955409.html

    #以下详解来自http://ailee.me/2018/05/15/Caffe-Python%E6%8E%A5%E5%8F%A3%E5%B8%B8%E7%94%A8API%E5%8F%82%E8%80%83/#21-data-layers-%E6%95%B0%E6%8D%AE%E5%B1%82%E5%AE%9A%E4%B9%89
################################################################################
# data,label:   为top的名称
# Data():       表示该层类型为数据层，数据来自于levelDB或者LMDB。
# source:       lmdb数据目录。
# backend:      选择是采用levelDB还是LMDB，默认是levelDB。
# batch_size:   每一次处理数据的个数。
# ntop:         表明有多少个blobs数据输出，示例中为2，代表着data和label。
# phase:        0:表示TRAIN
#               1:表示TEST
# transform_param:  数据预处理
#   scale:      归一化。1/255即将输入数据从0-255归一化到0-1之间。
#   crop_size:  对图像进行裁剪。如果定义了crop_size，那么在train时会对大
#               于crop_size的图片进行随机裁剪，而在test时只是截取中间部分。
#   mean_value: 图像通道的均值。三个值表示RGB图像中三个通道的均值。
#   mirror:     图像镜像。True为使用镜像。
################################################################################
#lmdb中存放了两个数据，data和label，因此在这个层上面会有两个blobs，所以ntop = 2
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

################################################################################
# bottom:       上一层数据输出。
# kernel_size:  卷积核大小。
# stride:       卷积核的步长，如果卷积核的长和宽不等，需要使用kernel_h和kernel_w
#               分别设定。
# num_output:   卷积核的数量。
# pad:          扩充边缘，默认为0，不扩充。扩充的时候上下、左右对称，比如卷积核为5*5，
#               那么pad设置为2，则在四个边缘都扩充两个像素，即宽和高都扩充4个像素，这
#               样卷积运算之后特征图不会变小。也可以使用pad_h和pad_w来分别设定。
# group:        分组，默认为1组。如果大于1，我们限制卷积的连接操作在一个子集内，如果
#               我们根据图像的通道来分组，那么第i个输出分组只能与第i个输入分组进行连接。
# weight_filler:权值初始化方式。默认为“constant”，值全为0，很多时候使用“xavier”算法
#               进行初始化。可选方式有：
#               constant:           常数初始化（默认为0）
#               gaussian:           高斯分布初试化权值
#               positive_unitball:  该方式可防止权值过大
#               uniform:            均匀分布初始化
#               xavier:             xavier算法初始化
#               msra:
#               billinear:          双线性插值初始化
# bias_filler:  偏置项初始化。一般设置为“constant”，值全为0.
# bias_term:    是否开启偏置项，默认为true，开启。
################################################################################
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))

################################################################################
# bottom:       上一层数据输出。
# pool:         池化方式，默认为MAX。目前可用的方法有MAX, AVE, 或STOCHASTIC。
# kernel_size:  池化的核大小。也可以用kernel_h和kernel_w分别设定。
# stride:       池化的步长，默认为1。一般我们设置为2，即不重叠。也可以用stride_h和
#               stride_w来设置。
################################################################################
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#全连接层，把输入当作成一个向量，输出也是一个简单向量（把输入数据blobs的width和height全变为1）。
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))

################################################################################
# negative_slope：  默认为0. 对标准的ReLU函数进行变化，如果设置了这个值，那么数据为
#                   负数时，就不再设置为0，而是用原始数据乘以negative_slope。
################################################################################
    n.relu1 = L.ReLU(n.fc1, in_place=True)

    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))

################################################################################
# 该层没有额外的参数。
# bottom:       数据输入（n.fc）
# bottom:       数据输入（n.label）
################################################################################
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()


#关于with as的用法见https://blog.csdn.net/u012609509/article/details/72911564
#在这里是用于检测异常
with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
    
with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

```

### 3 生成的文件

>以下是lenet_auto_train.prototxt的内容

```
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.00392156862745
  }
  data_param {
    source: "mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  convolution_param {
    num_output: 50
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "fc1"
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
layer {
  name: "score"
  type: "InnerProduct"
  bottom: "fc1"
  top: "score"
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
}

```
>以下是lenet_auto_solver.prototxt的内容
```
# 以下是train_net和test_net的文件定义
train_net: "mnist/lenet_auto_train.prototxt"
test_net: "mnist/lenet_auto_test.prototxt"
# test_iter 指定了通过test的数量
# 在这个MNIST的例子当中,我们设定100的batch的size以及100轮
# 覆盖完整的10,000个测试图像
test_iter: 100
# 每500次训练（train）之后进行一次测试
test_interval: 500
# 网络的基本学习速率（base_lr）、动量（momentum)以及网络的权重衰减（weight decay）
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# 学习策略
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# 每100轮展现一下
display: 100
# 轮数的最大值
max_iter: 10000
# 设置训练多少次之后保存一次快照
snapshot: 5000
# 快照位置
snapshot_prefix: "mnist/lenet"

```
### 4 加载和检查Solver

```Python
# -*- coding: utf-8 -*-
import caffe
from pylab import *

'''
本来是要使用GPU模式的，但是没有安装GPU相关组件
caffe.set_device(0)
caffe.set_mode_gpu()
'''
#所以只能设置为CPU模式

caffe.set_mode_cpu()

import sys
caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')

import os
os.chdir(caffe_root + '\\caffe\\examples')

#载入solver并且创建训练（train）和测试（test）网络
#加载solver文件，此处用SGD的方法来优化损失函数，当然也可以使用其他优化方法（比如Adagrad以及Nesterov的累计下降）
solver = None#忽略lmdb数据的这种解决方法（无法在同一数据上实例化两个求解器）
solver = caffe.SGDSolver('C:\\caffe\\caffe\\examples\\mnist\\lenet_auto_solver.prototxt')
#!!!如果出现Check failed:mdb_status == 0 （3 vs. 0）系统找不到指定的路径，是lmdb的位置写错了的原因


#为了解网络架构，我们可以检查中间的特征（blobs）以及参数（parameters）的维度

#每一个输出是（batch size, feature dim, spatial dim）
#产生列表:[(k, v.data.shape) for k,v in solver.net.blobs.items()]

print 'The shape of blobs:\n'

for k, v in solver.net.blobs.items():
    print k ,v.data.shape, '\n'


#输出权重（weight）的shape
#产生列表[(k,v[0].data.shape) for k,v in solver.net.params.items()]


print 'The shape of params:\n'
for k, v in solver.net.params.items():
    print k, v[0].data.shape, '\n'

#我们执行前馈过程（forward pass），并且测试网络

solver.net.forward()#训练网络
solver.test_nets[0].forward()#测试网络

#只取前8张图片，此时data[:8,0]的shape为(8,28,28),经过transpose变成(28,8,28)，camp指定颜色表为灰度值
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
show()

print 'train labels:', solver.net.blobs['label'].data[:8]

imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
show()
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]

#使用mini-batch梯度下降进行训练

'''
关于前馈函数solver.net.forward(), solver.test_nets[0].forward()和solver.step(1)的讲解见：
https://blog.csdn.net/u012762410/article/details/78917540

Blob的讲解：https://www.jianshu.com/p/0ac09c3ffec0
https://blog.csdn.net/jinxueliu31/article/details/52066709
http://yufeigan.github.io/2014/12/09/Caffe%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B02-Caffe%E7%9A%84%E4%B8%89%E7%BA%A7%E7%BB%93%E6%9E%84-Blobs-Layers-Nets/
'''

#训练1个batch看一下效果
#solver.step(1)是进行前馈并有反向传播的过程，1代表的是训练1次
#diff代表的是梯度信息（导数）

imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
show()

imshow(solver.net.params['conv1'][0].data[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
show()


```

```Python
# -*- coding: utf-8 -*-
import caffe
from lenet3 import *
from numpy import *
import time

#caffe中的python接口详解：https://www.cnblogs.com/yinheyi/p/6062488.html
#该example的中文介绍：https://buptldy.github.io/2016/05/05/2016-05-05-Caffe%20Python/
start = time.clock()

niter = 200
test_interval = 25

#losses will also be stored in the log

train_loss = zeros(niter)
test_acc = zeros(int(ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

#the main solver loop

for it in range(niter):
    #slover.step(1）表示训练1次，对训练网络进行一次正向与反向传播，并更新权值与偏置；
    #solver.solve()会进行完整的梯度训练，直至在solver中规定的max_iter
    solver.step(1)

    #存储每一轮的损失
    train_loss[it] = solver.net.blobs['loss'].data#一共200轮

    #存储第一个测试batch的输出(output)
    #从conv1层开始前馈避免传输新数据
    solver.test_nets[0].forward(start='conv1')
    #存储前8个的网络输出
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    #run a full test every so often每25轮运行一次测试
    #（Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)

    if it % test_interval == 0:#每25轮存储一次测试精度（第0、25、50、75、100、125、150、175）
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):#对测试集进行测试，100次，每一次训练一个batch-size（100个），这个100是测试集的batch
            solver.test_nets[0].forward()#关于argmax(axis)axis代表数轴，数轴为1，即压缩该轴，使该轴的维度变为1，比较其他轴最大的值的索引放在该轴上
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        #双斜杠（//）表示地板除，即先做除法（/），然后向下取整（floor）。至少有一方是float型时，结果为float型；两个数都是int型时，结果为int型。
        #http://qaru.site/questions/5405649/caffe-accuracy-bigger-than-100    
        test_acc[it // test_interval] = correct / 1e4#1e4 basically means 1*10^4 = batch_size * 迭代次数（100）

print 'Time used:', time.clock()-start
#matplotlib的基本概念讲解：https://www.cnblogs.com/nju2014/p/5620776.html
_, ax1 = subplots()#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html _代表是figure，ax1代表的是axes
ax2 = ax1.twinx()#创建一个与ax1共享x轴的实例（包含一个不可见的x轴）和一个与原始轴相对的独立y轴（右边），x轴的自动缩放设置将从原始轴继承，要确保两个y轴的刻度线对齐
ax1.plot(arange(niter), train_loss)#arange是numpy中的函数，用于创建等差数组，类似range，niter=200,创建的就是[0 1 2 ... 199]，train_loss存储着对应的每一轮的损失
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')#len(test_acc)计算test_acc的长度，在这里是8，见第40行，'r'代表颜色为红色
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
show()
```
效果如下![image](../Files%20about%20the%20installation%20of%20caffe/11.png)
