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

我们假设你已经成功编译了caffe。 如果没有，请参阅安装页面。 这个例子建立在MNIST教程智商，所以请在继续之前阅读它。

*指南指定所有路径并假设所有命令都从根caffe目录执行*
>
----------------------------------------
### 准备数据
切换到caffe根目录下<br />
运行get_mnist.sh文件<br />
```Linux
bash ./data/mnist/get_mnist/sh
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
首先，我们需要定义一个使用暹罗网络训练的模型。我们需要使用./examples/siamese/nist_siamese.prototxt。<br />
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
