### Introduction
While Caffe is made for deep networks it can likewise represent "shallow" models like logistic regression for classification. We'll do simple logistic regression on synthetic data that we'll generate and save to HDF5 to feed vectors to Caffe. Once that model is done, we'll add layers to improve accuracy. That's what Caffe is about: define a model, experiment, and then deploy.<br />
虽然Caffe是用于深度网络，但它同样可以表示“浅”模型，如分类的逻辑回归。 我们将对我们将生成的合成数据进行简单的逻辑回归，并保存到HDF5格式的数据集中，用以输入到Caffe中。 一旦该模型完成，我们将添加层以提高准确性。 <br />
这就是Caffe的意义：定义一个模型，实验，然后部署。<br />
>
Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative features and 2 noise features.<br />
创建一个10000*4的数据，用于二元分类，具有2个信息特征和2个噪声特征。

**如果在运行过程中，出现“module compiled against API version 0xc but this version of numpy is 0xb”类似的错误**<br />
解决办法：**升级numpy（注意我们这里使用的numpy版本是numpy+mkl版）到最新版**<br />

# -*- coding: utf-8 -*-
#一些其他资料：
#sklearn学习笔记（关于class_weight有详细的介绍）：http://www.bubuko.com/infodetail-2393539.html
#SGD介绍：https://blog.csdn.net/u010248552/article/details/79764340
#核密度估计资料：https://zhuanlan.zhihu.com/p/39962383

#关于该例子的翻译：https://blog.csdn.net/muyouhang/article/details/51078038
#关于该例子的分析：https://www.cnblogs.com/nwpuxuezha/p/4297853.html
#散步矩阵实例分析：https://blog.csdn.net/hurry0808/article/details/78573585?locationNum=7&fps=1

Brewing_logreg.py
```Python

import numpy as np
import matplotlib.pyplot as plt


import os
os.chdir('C:\\caffe')

caffe_root = 'C:\\caffe'

import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')
import caffe


import os
import h5py
import shutil
import tempfile
# 如果查找不到sklearn module的话，在cmd下输入conda install scikit-learn
import sklearn
import sklearn.datasets
import sklearn.linear_model

import pandas as pd

# 关于生成数据的介绍：https://blog.csdn.net/dataningwei/article/details/53649330
#通常用于分类算法。
#n_features :特征个数= n_informative（） + n_redundant + n_repeated
#n_informative：多信息特征的个数
#n_redundant：冗余信息，informative特征的随机线性组合
#n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
#n_classes：分类类别
#n_clusters_per_class ：某一个类别是由几个cluster构成的

'''
sklearn.datasets.make_classification产生测试数据。
10000组数据，特征向量维数为4。
sklearn.cross_validation.train_test_split为交叉验证。就是把data拆分为不同的train set和test set。
这里拆分为7500：2500

'''


X, y = sklearn.datasets.make_classification(
    n_samples=10000, n_features=4, n_redundant=0, n_informative=2, 
    n_clusters_per_class=2, hypercube=False, random_state=0
)

# 随机划分训练集和测试集

#train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata
#官方文档：http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html

from sklearn.model_selection import train_test_split
X, Xt, y, yt = train_test_split(X, y)

# Visualize sample of the data
# 随机生成一个序列或者返回一个置换的范围
# X.shape=7500
# 在此产生0-7499乱序序列并取前1000
ind = np.random.permutation(X.shape[0])[:1000]

#DataFrame 类型类似于数据库表结构的数据结构，其含有行索引和列索引，
#可以将DataFrame 想成是由相同索引的Series组成的Dict类型。在其底层是通过二维以及一维的数据块实现。

#详细介绍：https://www.cnblogs.com/linux-wangkun/p/5903945.html
df = pd.DataFrame(X[ind])
# 产生散点图，介绍文档：https://blog.csdn.net/hurry0808/article/details/78573585?locationNum=7&fps=1
# scatter_matrix的应用：https://www.jianshu.com/p/9a2a0a4c76dd

_ = pd.scatter_matrix(df, figsize=(9, 9), diagonal='kde', marker='o', s=40, alpha=.4, c=y[ind])
# df：数据名称
# alpha：透明度
# diagonal：hist(直方图）和kde（核密度估计）
# s：散点的大小
# c：散点的颜色
# vmin，vmax：高度设置，标量
# cmap：colormap
#关于kde---核密度估计的介绍：https://zhuanlan.zhihu.com/p/39962383
plt.show()


# 训练，并且测试SGD逻辑回归的性能
clf = sklearn.linear_model.SGDClassifier(
    loss='log', n_iter=1000, penalty='l2', alpha=5e-4, class_weight='balanced')
#loss:损失函数类型
#n_iter:达到停止标准的实际迭代次数
#penalty:正则化类型
#alpha:用于被常数乘的正则项，默认是0.001，设置为最优时，被用于计算learning_rate
#calss_weight:每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
clf.fit(X, y)# 用训练数据拟合分类器模型
yt_pred = clf.predict(Xt)
print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))


# 将数据集保存到HDF5文件中，以加载到Caffe中。

# Write out the data to HDF5 files in a temp directory.
# This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
dirname = os.path.abspath(caffe_root+'\\caffe\\examples\\hdf5_classification\\data')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname, 'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)#float32是深度学习非常常见的一种类型，关于float32的介绍https://blog.csdn.net/lien0906/article/details/78863118
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)##双星号可以用来获得字典的值
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')


```

BrewToCaffe.py
```Python
# -*- coding: utf-8 -*-
from Brewing_logreg import *
import numpy as np
import matplotlib.pyplot as plt


import os
os.chdir('C:\\caffe')

caffe_root = 'C:\\caffe'

import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')
import caffe


import os
import h5py
import shutil
import tempfile
# 如果查找不到sklearn module的话，在cmd下输入conda install scikit-learn
import sklearn
import sklearn.datasets
import sklearn.linear_model

#import pandas as pd

from caffe import layers as L
from caffe import params as P


#让我们通过Python网络规范在Caffe中定义逻辑回归。 这是一个快速和自然的方式来定义网络，旁边手动编辑protobuf模型。
def logreg(hdf5, batch_size):
    # logistic regression: data, matrix multiplication, and 2-class softmax loss
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip1, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()

train_net_path = 'C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(logreg('C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\train.txt', 10)))

test_net_path = 'C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(logreg('C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\test.txt', 10)))

#现在，我们将定义我们的“solver”，通过指定上面定义的训练和测试网络的位置用于学习，显示和“快照”的各种参数的值来训练网络。

from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 1000  # Test after every 1000 training iterations.
    s.test_iter.append(250) # Test 250 "batches" each time we test.

    s.max_iter = 10000      # # of times to update the net (training iterations)

    # Set the initial learning rate for stochastic gradient descent (SGD).
    s.base_lr = 0.01        

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000

    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = 10000
    s.snapshot_prefix = 'C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\train'

    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    return s

solver_path = 'C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))

#现在应该是时候去学习和评估我们的Caffeinated逻辑回归，在Python中。

caffe.set_mode_cpu()
solver = caffe.get_solver(solver_path)
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(Xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print("Accuracy: {:.3f}".format(accuracy))
#通过命令行界面做同样的操作，以获得模型和求解的详细输出。
```
接下来训练模型，进入cygwin
```

沙@DESKTOP-RMK94TQ /cygdrive/c/caffe/caffe
$ ./scripts/build/tools/Release/caffe train --solver ./examples/hdf5_classification/logreg_solver.prototxt
I0820 17:58:59.517169 51668 caffe.cpp:212] Use CPU.
I0820 17:58:59.518167 51668 solver.cpp:44] Initializing solver from parameters:
train_net: "C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_train.prototxt"
test_net: "C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_test.prototxt"
test_iter: 250
test_interval: 1000
base_lr: 0.01
display: 1000
max_iter: 10000
lr_policy: "step"
gamma: 0.1
momentum: 0.9
weight_decay: 0.0005
stepsize: 5000
snapshot: 10000
snapshot_prefix: "C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\train"
solver_mode: CPU
train_state {
  level: 0
  stage: ""
}
I0820 17:58:59.519165 51668 solver.cpp:77] Creating training net from train_net                                                                                                                                                                                                file: C:\caffe\caffe\examples\hdf5_classification\logreg_auto_train.prototxt
I0820 17:58:59.519165 51668 net.cpp:51] Initializing net from parameters:
state {
  phase: TRAIN
  level: 0
  stage: ""
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\train.txt"
    batch_size: 10
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip1"
  bottom: "label"
  top: "accuracy"
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip1"
  bottom: "label"
  top: "loss"
}
I0820 17:58:59.519665 51668 layer_factory.cpp:58] Creating layer data
I0820 17:58:59.519665 51668 net.cpp:84] Creating Layer data
I0820 17:58:59.519665 51668 net.cpp:380] data -> data
I0820 17:58:59.519665 51668 net.cpp:380] data -> label
I0820 17:58:59.519665 51668 hdf5_data_layer.cpp:80] Loading list of HDF5 filenamesfrom: C:\caffe\caffe\examples\hdf5_classification\data\train.txt
I0820 17:58:59.519665 51668 hdf5_data_layer.cpp:94] Number of HDF5 files: 2
I0820 17:58:59.521162 51668 hdf5.cpp:32] Datatype class: H5T_FLOAT
I0820 17:58:59.521661 51668 net.cpp:122] Setting up data
I0820 17:58:59.521661 51668 net.cpp:129] Top shape: 10 4 (40)
I0820 17:58:59.521661 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.521661 51668 net.cpp:137] Memory required for data: 200
I0820 17:58:59.521661 51668 layer_factory.cpp:58] Creating layer label_data_1_split
I0820 17:58:59.521661 51668 net.cpp:84] Creating Layer label_data_1_split
I0820 17:58:59.521661 51668 net.cpp:406] label_data_1_split <- label
I0820 17:58:59.522159 51668 net.cpp:380] label_data_1_split -> label_data_1_split_0
I0820 17:58:59.522159 51668 net.cpp:380] label_data_1_split -> label_data_1_split_1
I0820 17:58:59.522159 51668 net.cpp:122] Setting up label_data_1_split
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.522159 51668 net.cpp:137] Memory required for data: 280
I0820 17:58:59.522159 51668 layer_factory.cpp:58] Creating layer ip1
I0820 17:58:59.522159 51668 net.cpp:84] Creating Layer ip1
I0820 17:58:59.522159 51668 net.cpp:406] ip1 <- data
I0820 17:58:59.522159 51668 net.cpp:380] ip1 -> ip1
I0820 17:58:59.522159 51668 common.cpp:36] System entropy source not available,using fallback algorithm to generate seed instead.
I0820 17:58:59.522159 51668 net.cpp:122] Setting up ip1
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.522159 51668 net.cpp:137] Memory required for data: 360
I0820 17:58:59.522159 51668 layer_factory.cpp:58] Creating layer ip1_ip1_0_split
I0820 17:58:59.522159 51668 net.cpp:84] Creating Layer ip1_ip1_0_split
I0820 17:58:59.522159 51668 net.cpp:406] ip1_ip1_0_split <- ip1
I0820 17:58:59.522159 51668 net.cpp:380] ip1_ip1_0_split -> ip1_ip1_0_split_0
I0820 17:58:59.522159 51668 net.cpp:380] ip1_ip1_0_split -> ip1_ip1_0_split_1
I0820 17:58:59.522159 51668 net.cpp:122] Setting up ip1_ip1_0_split
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.522658 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.522658 51668 net.cpp:137] Memory required for data: 520
I0820 17:58:59.522658 51668 layer_factory.cpp:58] Creating layer accuracy
I0820 17:58:59.522658 51668 net.cpp:84] Creating Layer accuracy
I0820 17:58:59.522658 51668 net.cpp:406] accuracy <- ip1_ip1_0_split_0
I0820 17:58:59.522658 51668 net.cpp:406] accuracy <- label_data_1_split_0
I0820 17:58:59.522658 51668 net.cpp:380] accuracy -> accuracy
I0820 17:58:59.522658 51668 net.cpp:122] Setting up accuracy
I0820 17:58:59.522658 51668 net.cpp:129] Top shape: (1)
I0820 17:58:59.522658 51668 net.cpp:137] Memory required for data: 524
I0820 17:58:59.522658 51668 layer_factory.cpp:58] Creating layer loss
I0820 17:58:59.522658 51668 net.cpp:84] Creating Layer loss
I0820 17:58:59.523157 51668 net.cpp:406] loss <- ip1_ip1_0_split_1
I0820 17:58:59.523157 51668 net.cpp:406] loss <- label_data_1_split_1
I0820 17:58:59.523157 51668 net.cpp:380] loss -> loss
I0820 17:58:59.523157 51668 layer_factory.cpp:58] Creating layer loss
I0820 17:58:59.523157 51668 net.cpp:122] Setting up loss
I0820 17:58:59.523157 51668 net.cpp:129] Top shape: (1)
I0820 17:58:59.523157 51668 net.cpp:132]     with loss weight 1
I0820 17:58:59.523157 51668 net.cpp:137] Memory required for data: 528
I0820 17:58:59.523157 51668 net.cpp:198] loss needs backward computation.
I0820 17:58:59.523157 51668 net.cpp:200] accuracy does not need backward computation.
I0820 17:58:59.523157 51668 net.cpp:198] ip1_ip1_0_split needs backward computation.
I0820 17:58:59.523157 51668 net.cpp:198] ip1 needs backward computation.
I0820 17:58:59.523157 51668 net.cpp:200] label_data_1_split does not need backwardcomputation.
I0820 17:58:59.523157 51668 net.cpp:200] data does not need backward computationI0820 17:58:59.523157 51668 net.cpp:242] This network produces output accuracy
I0820 17:58:59.523157 51668 net.cpp:242] This network produces output loss
I0820 17:58:59.523157 51668 net.cpp:255] Network initialization done.
I0820 17:58:59.523656 51668 solver.cpp:172] Creating test net (#0) specified bytest_net file: C:\caffe\caffe\examples\hdf5_classification\logreg_auto_test.prototxt
I0820 17:58:59.523656 51668 net.cpp:51] Initializing net from parameters:
state {
  phase: TEST
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\test.txt"
    batch_size: 10
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip1"
  bottom: "label"
  top: "accuracy"
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip1"
  bottom: "label"
  top: "loss"
}
I0820 17:58:59.523656 51668 layer_factory.cpp:58] Creating layer data
I0820 17:58:59.523656 51668 net.cpp:84] Creating Layer data
I0820 17:58:59.523656 51668 net.cpp:380] data -> data
I0820 17:58:59.523656 51668 net.cpp:380] data -> label
I0820 17:58:59.523656 51668 hdf5_data_layer.cpp:80] Loading list of HDF5 filenamesfrom: C:\caffe\caffe\examples\hdf5_classification\data\test.txt
I0820 17:58:59.524155 51668 hdf5_data_layer.cpp:94] Number of HDF5 files: 1
I0820 17:58:59.525652 51668 net.cpp:122] Setting up data
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 4 (40)
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.525652 51668 net.cpp:137] Memory required for data: 200
I0820 17:58:59.525652 51668 layer_factory.cpp:58] Creating layer label_data_1_split
I0820 17:58:59.525652 51668 net.cpp:84] Creating Layer label_data_1_split
I0820 17:58:59.525652 51668 net.cpp:406] label_data_1_split <- label
I0820 17:58:59.525652 51668 net.cpp:380] label_data_1_split -> label_data_1_split_0
I0820 17:58:59.525652 51668 net.cpp:380] label_data_1_split -> label_data_1_split_1
I0820 17:58:59.525652 51668 net.cpp:122] Setting up label_data_1_split
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.525652 51668 net.cpp:137] Memory required for data: 280
I0820 17:58:59.525652 51668 layer_factory.cpp:58] Creating layer ip1
I0820 17:58:59.525652 51668 net.cpp:84] Creating Layer ip1
I0820 17:58:59.525652 51668 net.cpp:406] ip1 <- data
I0820 17:58:59.525652 51668 net.cpp:380] ip1 -> ip1
I0820 17:58:59.525652 51668 net.cpp:122] Setting up ip1
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.525652 51668 net.cpp:137] Memory required for data: 360
I0820 17:58:59.525652 51668 layer_factory.cpp:58] Creating layer ip1_ip1_0_split
I0820 17:58:59.525652 51668 net.cpp:84] Creating Layer ip1_ip1_0_split
I0820 17:58:59.526151 51668 net.cpp:406] ip1_ip1_0_split <- ip1
I0820 17:58:59.526151 51668 net.cpp:380] ip1_ip1_0_split -> ip1_ip1_0_split_0
I0820 17:58:59.526151 51668 net.cpp:380] ip1_ip1_0_split -> ip1_ip1_0_split_1
I0820 17:58:59.526151 51668 net.cpp:122] Setting up ip1_ip1_0_split
I0820 17:58:59.526151 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.526151 51668 net.cpp:129] Top shape: 10 2 (20)
I0820 17:58:59.526151 51668 net.cpp:137] Memory required for data: 520
I0820 17:58:59.526151 51668 layer_factory.cpp:58] Creating layer accuracy
I0820 17:58:59.526151 51668 net.cpp:84] Creating Layer accuracy
I0820 17:58:59.526151 51668 net.cpp:406] accuracy <- ip1_ip1_0_split_0
I0820 17:58:59.526151 51668 net.cpp:406] accuracy <- label_data_1_split_0
I0820 17:58:59.526151 51668 net.cpp:380] accuracy -> accuracy
I0820 17:58:59.526151 51668 net.cpp:122] Setting up accuracy
I0820 17:58:59.526151 51668 net.cpp:129] Top shape: (1)
I0820 17:58:59.526151 51668 net.cpp:137] Memory required for data: 524
I0820 17:58:59.526151 51668 layer_factory.cpp:58] Creating layer loss
I0820 17:58:59.526151 51668 net.cpp:84] Creating Layer loss
I0820 17:58:59.526151 51668 net.cpp:406] loss <- ip1_ip1_0_split_1
I0820 17:58:59.526151 51668 net.cpp:406] loss <- label_data_1_split_1
I0820 17:58:59.526151 51668 net.cpp:380] loss -> loss
I0820 17:58:59.526151 51668 layer_factory.cpp:58] Creating layer loss
I0820 17:58:59.526151 51668 net.cpp:122] Setting up loss
I0820 17:58:59.526151 51668 net.cpp:129] Top shape: (1)
I0820 17:58:59.526151 51668 net.cpp:132]     with loss weight 1
I0820 17:58:59.526151 51668 net.cpp:137] Memory required for data: 528
I0820 17:58:59.526151 51668 net.cpp:198] loss needs backward computation.
I0820 17:58:59.526151 51668 net.cpp:200] accuracy does not need backward computation.
I0820 17:58:59.526151 51668 net.cpp:198] ip1_ip1_0_split needs backward computation.
I0820 17:58:59.526151 51668 net.cpp:198] ip1 needs backward computation.
I0820 17:58:59.526151 51668 net.cpp:200] label_data_1_split does not need backward computation.
I0820 17:58:59.526151 51668 net.cpp:200] data does not need backward computation.
I0820 17:58:59.526151 51668 net.cpp:242] This network produces output accuracy
I0820 17:58:59.526151 51668 net.cpp:242] This network produces output loss
I0820 17:58:59.526151 51668 net.cpp:255] Network initialization done.
I0820 17:58:59.526650 51668 solver.cpp:56] Solver scaffolding done.
I0820 17:58:59.526650 51668 caffe.cpp:249] Starting Optimization
I0820 17:58:59.526650 51668 solver.cpp:272] Solving
I0820 17:58:59.526650 51668 solver.cpp:273] Learning Rate Policy: step
I0820 17:58:59.526650 51668 solver.cpp:330] Iteration 0, Testing net (#0)
I0820 17:58:59.530144 51668 solver.cpp:397]     Test net output #0: accuracy = 0.5124
I0820 17:58:59.530144 51668 solver.cpp:397]     Test net output #1: loss = 0.808089 (* 1 = 0.808089 loss)
I0820 17:58:59.530144 51668 solver.cpp:218] Iteration 0 (0 iter/s, 0.003s/1000 iters), loss = 0.87678
I0820 17:58:59.530144 51668 solver.cpp:237]     Train net output #0: accuracy =0.5
I0820 17:58:59.530144 51668 solver.cpp:237]     Train net output #1: loss = 0.87678 (* 1 = 0.87678 loss)
I0820 17:58:59.530144 51668 sgd_solver.cpp:105] Iteration 0, lr = 0.01
I0820 17:58:59.541124 51668 solver.cpp:330] Iteration 1000, Testing net (#0)
I0820 17:58:59.543619 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7644
I0820 17:58:59.543619 51668 solver.cpp:397]     Test net output #1: loss = 0.59192 (* 1 = 0.59192 loss)
I0820 17:58:59.543619 51668 solver.cpp:218] Iteration 1000 (76923.1 iter/s, 0.013s/1000 iters), loss = 0.547078
I0820 17:58:59.543619 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.543619 51668 solver.cpp:237]     Train net output #1: loss = 0.547078 (* 1 = 0.547078 loss)
I0820 17:58:59.543619 51668 sgd_solver.cpp:105] Iteration 1000, lr = 0.01
I0820 17:58:59.556095 51668 solver.cpp:330] Iteration 2000, Testing net (#0)
I0820 17:58:59.558092 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7532
I0820 17:58:59.558092 51668 solver.cpp:397]     Test net output #1: loss = 0.587112 (* 1 = 0.587112 loss)
I0820 17:58:59.558092 51668 solver.cpp:218] Iteration 2000 (71428.6 iter/s, 0.014s/1000 iters), loss = 0.644126
I0820 17:58:59.558092 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.558092 51668 solver.cpp:237]     Train net output #1: loss = 0.644126 (* 1 = 0.644126 loss)
I0820 17:58:59.558092 51668 sgd_solver.cpp:105] Iteration 2000, lr = 0.01
I0820 17:58:59.570567 51668 solver.cpp:330] Iteration 3000, Testing net (#0)
I0820 17:58:59.573063 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7572
I0820 17:58:59.573063 51668 solver.cpp:397]     Test net output #1: loss = 0.593993 (* 1 = 0.593993 loss)
I0820 17:58:59.573063 51668 solver.cpp:218] Iteration 3000 (71428.6 iter/s, 0.014s/1000 iters), loss = 0.57215
I0820 17:58:59.573063 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.573063 51668 solver.cpp:237]     Train net output #1: loss = 0.57215 (* 1 = 0.57215 loss)
I0820 17:58:59.573063 51668 sgd_solver.cpp:105] Iteration 3000, lr = 0.01
I0820 17:58:59.585541 51668 solver.cpp:330] Iteration 4000, Testing net (#0)
I0820 17:58:59.587536 51668 solver.cpp:397]     Test net output #0: accuracy = 0 .7644
I0820 17:58:59.587536 51668 solver.cpp:397]     Test net output #1: loss = 0.59192 (* 1 = 0.59192 loss)
I0820 17:58:59.587536 51668 solver.cpp:218] Iteration 4000 (71428.6 iter/s, 0.014s/1000 iters), loss = 0.547078
I0820 17:58:59.587536 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.587536 51668 solver.cpp:237]     Train net output #1: loss = 0.547078 (* 1 = 0.547078 loss)
I0820 17:58:59.587536 51668 sgd_solver.cpp:105] Iteration 4000, lr = 0.01
I0820 17:58:59.599511 51668 solver.cpp:330] Iteration 5000, Testing net (#0)
I0820 17:58:59.601508 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7532
I0820 17:58:59.601508 51668 solver.cpp:397]     Test net output #1: loss = 0.587112 (* 1 = 0.587112 loss)
I0820 17:58:59.601508 51668 solver.cpp:218] Iteration 5000 (76923.1 iter/s, 0.013s/1000 iters), loss = 0.644126
I0820 17:58:59.601508 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.601508 51668 solver.cpp:237]     Train net output #1: loss = 0.644126 (* 1 = 0.644126 loss)
I0820 17:58:59.602012 51668 sgd_solver.cpp:105] Iteration 5000, lr = 0.001
I0820 17:58:59.614982 51668 solver.cpp:330] Iteration 6000, Testing net (#0)
I0820 17:58:59.616979 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7776
I0820 17:58:59.616979 51668 solver.cpp:397]     Test net output #1: loss = 0.588383 (* 1 = 0.588383 loss)
I0820 17:58:59.616979 51668 solver.cpp:218] Iteration 6000 (66666.7 iter/s, 0.015s/1000 iters), loss = 0.583577
I0820 17:58:59.616979 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.616979 51668 solver.cpp:237]     Train net output #1: loss = 0.583577 (* 1 = 0.583577 loss)
I0820 17:58:59.616979 51668 sgd_solver.cpp:105] Iteration 6000, lr = 0.001
I0820 17:58:59.628458 51668 solver.cpp:330] Iteration 7000, Testing net (#0)
I0820 17:58:59.630455 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7792
I0820 17:58:59.630954 51668 solver.cpp:397]     Test net output #1: loss = 0.588225 (* 1 = 0.588225 loss)
I0820 17:58:59.630954 51668 solver.cpp:218] Iteration 7000 (76923.1 iter/s, 0.013s/1000 iters), loss = 0.5487
I0820 17:58:59.630954 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.630954 51668 solver.cpp:237]     Train net output #1: loss = 0.54(* 1 = 0.5487 loss)
I0820 17:58:59.630954 51668 sgd_solver.cpp:105] Iteration 7000, lr = 0.001
I0820 17:58:59.641932 51668 solver.cpp:330] Iteration 8000, Testing net (#0)
I0820 17:58:59.644428 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7772
I0820 17:58:59.644428 51668 solver.cpp:397]     Test net output #1: loss = 0.587423 (* 1 = 0.587423 loss)
I0820 17:58:59.644428 51668 solver.cpp:218] Iteration 8000 (76923.1 iter/s, 0.013s/1000 iters), loss = 0.670083
I0820 17:58:59.644925 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.644925 51668 solver.cpp:237]     Train net output #1: loss = 0.670083 (* 1 = 0.670083 loss)
I0820 17:58:59.644925 51668 sgd_solver.cpp:105] Iteration 8000, lr = 0.001
I0820 17:58:59.656404 51668 solver.cpp:330] Iteration 9000, Testing net (#0)
I0820 17:58:59.659397 51668 solver.cpp:397]     Test net output #0: accuracy = 0.7784
I0820 17:58:59.659397 51668 solver.cpp:397]     Test net output #1: loss = 0.588682 (* 1 = 0.588682 loss)
I0820 17:58:59.659397 51668 solver.cpp:218] Iteration 9000 (71428.6 iter/s, 0.014s/1000 iters), loss = 0.580133
I0820 17:58:59.659397 51668 solver.cpp:237]     Train net output #0: accuracy = 0.7
I0820 17:58:59.659397 51668 solver.cpp:237]     Train net output #1: loss = 0.580133 (* 1 = 0.580133 loss)
I0820 17:58:59.659898 51668 sgd_solver.cpp:105] Iteration 9000, lr = 0.001
I0820 17:58:59.670877 51668 solver.cpp:447] Snapshotting to binary proto file C:\caffe\caffe\examples\hdf5_classification\data\train_iter_10000.caffemodel
I0820 17:58:59.671376 51668 sgd_solver.cpp:273] Snapshotting solver state to binary proto file C:\caffe\caffe\examples\hdf5_classification\data\train_iter_10000.solverstate
I0820 17:58:59.671875 51668 solver.cpp:310] Iteration 10000, loss = 0.548088
I0820 17:58:59.671875 51668 solver.cpp:330] Iteration 10000, Testing net (#0)
I0820 17:58:59.673871 51668 solver.cpp:397]     Test net output #0: accuracy = 0.78
I0820 17:58:59.673871 51668 solver.cpp:397]     Test net output #1: loss = 0.588317 (* 1 = 0.588317 loss)
I0820 17:58:59.673871 51668 solver.cpp:315] Optimization Done.
I0820 17:58:59.673871 51668 caffe.cpp:260] Optimization Done.
```

这个模型的结构如下<br />
![log_test.png](Files%20about%20the%20installation%20of%20caffe/log_test.png)

NonLinearNet.py
```Python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Brewing_logreg import *

import os
os.chdir('C:\\caffe')

caffe_root = 'C:\\caffe'

import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')
import caffe


import os
import h5py
import shutil
import tempfile
# 如果查找不到sklearn module的话，在cmd下输入conda install scikit-learn
import sklearn
import sklearn.datasets
import sklearn.linear_model


from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 1000  # Test after every 1000 training iterations.
    s.test_iter.append(250) # Test 250 "batches" each time we test.

    s.max_iter = 10000      # # of times to update the net (training iterations)

    # Set the initial learning rate for stochastic gradient descent (SGD).
    s.base_lr = 0.01        

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000

    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = 10000
    s.snapshot_prefix = 'C:\\caffe\\caffe\\examples\\hdf5_classification\\data\\train'

    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    return s

def nonlinear_net(hdf5, batch_size):
    # one small nonlinearity, one leap for model kind
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    # define a hidden layer of dimension 40
    n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
    # transform the output through the ReLU (rectified linear) non-linearity
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # score the (now non-linear) features
    n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
    # same accuracy and loss as before
    n.accuracy = L.Accuracy(n.ip2, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

train_net_path = caffe_root + '\\caffe\\examples\\hdf5_classification\\nonlinear_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(nonlinear_net(caffe_root+'\\caffe\\examples\\hdf5_classification\\data\\train.txt', 10)))

test_net_path = caffe_root + '\\caffe\\examples\\hdf5_classification\\nonlinear_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(nonlinear_net(caffe_root+'\\caffe\\examples\\hdf5_classification\\data\\test.txt', 10)))

solver_path = caffe_root + '\\caffe\\examples\\hdf5_classification\\nonlinear_logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))

caffe.set_mode_cpu()
solver = caffe.get_solver(solver_path)
solver.solve()
#solver.solve()会进行完整的梯度训练，直至在solver中规定的max_iter

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(Xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print("Accuracy: {:.3f}".format(accuracy))

#shutil.rmtree(dirname)  递归删除一个目录以及目录内的所有内容
```
