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
