标题：暹罗网络教程<br />
描述：在MNIST数据上训练和测试暹罗网络。<br />
类别：示例<br />
include_in_docs：true<br />
布局：默认<br />
优先级：100<br />

使用Caffe进行暹罗网络培训<br />
>
---------------------------------------
这个Example显示了如何使用权重共享（sharing）和对比度损失函数（contrastive loss function）在Caffe中使用暹罗网络学习模型。

我们假设你已经成功编译了caffe。 如果没有，请参阅安装页面。 这个例子建立在MNIST教程之上，所以请在继续之前阅读它。

*假设所有命令都从根caffe目录执行*
>
----------------------------------------
### 准备数据
切换到caffe根目录下<br />
运行get_mnist.sh文件<br />
```Linux
bash ./data/mnist/get_mnist.sh
```
由于运行create_mnist_siamese.sh出现问题，所以需要修改一下，改为如下<br />
```
#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
set -e

EXAMPLES=./scripts/build/examples/siamese
DATA=./data/mnist

echo "Creating leveldb..."

rm -rf ./examples/siamese/mnist_siamese_train_leveldb
rm -rf ./examples/siamese/mnist_siamese_test_leveldb
#\ 表示换行（在LINUX系统中），在Windows系统中，用^来表示换行
$EXAMPLES/Release/convert_mnist_siamese_data.exe \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_train_leveldb
$EXAMPLES/Release/convert_mnist_siamese_data.exe \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_test_leveldb

echo "Done."

```
----------------------------------
### 模型
首先，我们需要定义一个使用暹罗网络训练的模型。我们需要使用./examples/siamese/mnist_siamese.prototxt。<br />
这个模型几乎与LeNet model是一样的,唯一的差异是，我们用产生二维向量的 linear "feature" layer替换了10个数字输出的类<br />
```
layer {
  name: "feat"
  type: "InnerProduct"
  bottom: "ip2"
  top: "feat"
  param {
    name: "feat_w"
    lr_mult: 1
  }
  param {
    name: "feat_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
  }
}
```
---------------------------------
### 定义Siamese网络
这个网络定义在./examples/siamese/mnist_siamese_train_test.prototxt。
>
-----------------------------------
### 从一对数据中读取数据（Reading in the Pair Data)
我们从一个数据层开始，该数据层从我们之前创建的LevelDB数据库中读取。 <br />
此数据库中的每个条目都包含一对图像（pair_data）的图像数据和一个二进制标签，表示它们属于同一个类还是不同的类（sim）。
>
--------------------------------------
```
layer {
  name: "pair_data"
  type: "Data"
  top: "pair_data"
  top: "sim"
  include { phase: TRAIN }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/siamese/mnist_siamese_train_leveldb"
    batch_size: 64
  }
}
```
----------------------------------------
为了将一对图像打包到数据库中的同一个blob中，我们为每个通道打包一个图像。<br />
我们希望能够分别处理这两个图像，因此在数据层之后添加一个切片层(slice layer)。<br />
它获取pair_data，并沿着通道维度对其进行切片，这样数据中就有一个单独的图像，data_p中就有一个成对的图像。
```
layer {
  name: "slice_pair"
  type: "Slice"
  bottom: "pair_data"
  top: "data"
  top: "data_p"
  slice_param {
    slice_dim: 1
    slice_point: 1
  }
}
```
---------------------------------------------
### 建立暹罗网络的First Side
我们现在可以指定Siamese Net的First Side。这个Side在数据上操作并且产生feat（produces feat）。从网络开始，<br />
从./example/siamese/mnist_siamese.prototxt我添加了默认的权重初始参数。然后我们命名卷基层和全连接层的参数。<br />
命名参数允许Caffe可以在Siamese网络两遍的层之间共享参数。在定义中，我们可以看到：<br />
```
...
param { name: "conv1_w" ...  }
param { name: "conv1_b" ...  }
...
param { name: "conv2_w" ...  }
param { name: "conv2_b" ...  }
...
param { name: "ip1_w" ...  }
param { name: "ip1_b" ...  }
...
param { name: "ip2_w" ...  }
param { name: "ip2_b" ...  }
...
```
---------------------------------------
### 暹罗网络的Second Side的建立
我们现在需要在data_p上建立第二个路径，并且产生feat_p。 这条路径与第一条完全一样。<br />
我们可以复制粘贴。<br />
然后，我们通过添加“_p”来改变每一层，输入和输出，以区分“paired” layers和原始层。<br />
>
--------------------------------------------
### 添加Contrastive Loss Function
为了训练网络，需要优化Contrastive loss函数，这个方法由Raia Hadsell, Sumit Chopra, Yann LeCun提出。<br />
“Dimensionality Reduction by Learning an Invariant Mapping”。  这种损失函数鼓励matching pairs在feature<br />
space中更加接近，而non-matching pairs会被分开。这个代价函数是通过CONTRASTIVE_LOSS layer实现的。<br />
```
layer {
    name: "loss"
    type: "ContrastiveLoss"
    contrastive_loss_param {
        margin: 1.0
    }
    bottom: "feat"
    bottom: "feat_p"
    bottom: "sim"
    top: "loss"
}
```
>
--------------------------------------------
### 定义Solver
我们需要将solver指向一个正确的model file。solver被定义在
```
./examples/siamese/mnist_siamese_solver.prototxt
```
-------------------------------------------------
### 训练和测试模型
!!!!!!!!!!!!!!!!!!!!!!!!高亮!!!!!!!!!!!!!!!!!!!!!!!!!!!!
由于一些不可知的原因，需要更改一下train_mnist_siamese.sh的内容<br />
如下<br />
```
#!/usr/bin/env sh
set -e

TOOLS=./scripts/build/tools

$TOOLS/Release/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt $@

```
除此之外，还需要修改一下mnist_siamese_solver.prototxt,将其改为CPU模式
运行<br />
```
bash ./examples/siamese/train_mnist_siamese.sh
```
-------------------------------------------------
### 描绘结果
首先，我们可以通过运行以下命令来绘制.prototxt文件中定义的DAGs。<br />
以此来描绘模型和暹罗网络
>
---------------------------------------------------
```
./python/draw_net.py \
    ./examples/siamese/mnist_siamese.prototxt \
    ./examples/siamese/mnist_siamese.png

./python/draw_net.py \
    ./examples/siamese/mnist_siamese_train_test.prototxt \
    ./examples/siamese/mnist_siamese_train_test.png
```

代码：
```
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

caffe_root = 'C:\\caffe'
import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')

import caffe

MODEL_FILE = caffe_root + '\\caffe\\examples\\siamese\\mnist_siamese.prototxt'
PRETRAINED_FILE = caffe_root + '\\caffe\\examples\\siamese\\mnist_siamese_iter_50000.caffemodel' 
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

TEST_DATA_FILE = caffe_root + '\\caffe\\data\\mnist\\t10k-images-idx3-ubyte'
TEST_LABEL_FILE = caffe_root + '\\caffe\\data\\mnist\\t10k-labels-idx1-ubyte'
n = 10000

with open(TEST_DATA_FILE, 'rb') as f: #rb代表读取二进制文件，关于with open as 的介绍：https://www.cnblogs.com/ymjyqsx/p/6554817.html
    f.read(16) # 每次最多读取16个字节的内容(读取前16个字节作为文件的头，不用于测试） //read([size])方法从文件当前位置起读取size个字节，若无参数size，则表示读取至文件结束为止，它范围为字符串对象，
    raw_data = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    #图像通常被编码成无符号八位整数（uint8），关于python的图像处理的介绍：https://blog.csdn.net/wuxiaobingandbob/article/details/51751899

with open(TEST_LABEL_FILE, 'rb') as f:
    f.read(8) # skip the header
    labels = np.fromstring(f.read(n), dtype=np.uint8)

# reshape and preprocess
caffe_in = raw_data.reshape(n, 1, 28, 28) * 0.00390625 # 实际上就是1/255manually scale data instead of using `caffe.io.Transformer`
out = net.forward_all(data=caffe_in)#批量前向传播

#https://blog.csdn.net/lilai619/article/details/54425157 pycaffe 接口介绍

feat = out['feat']
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    a = feat[labels==i, 0].flatten()
    b = feat[labels==i, 1].flatten()
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
    #sim = 1表示相似
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()
```

训练的结构图如下：<br />
<img src="Files%20about%20the%20installation%20of%20caffe/mynetwrk_TB.png" width = 70% div align="center"/>
测试的结构图如下：<br />
<img src="Files%20about%20the%20installation%20of%20caffe/Siamese_TB.png" width = 40% div align="center"/>
