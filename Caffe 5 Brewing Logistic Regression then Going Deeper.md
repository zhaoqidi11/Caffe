### Introduction
While Caffe is made for deep networks it can likewise represent "shallow" models like logistic regression for classification. We'll do simple logistic regression on synthetic data that we'll generate and save to HDF5 to feed vectors to Caffe. Once that model is done, we'll add layers to improve accuracy. That's what Caffe is about: define a model, experiment, and then deploy.<br />
虽然Caffe是用于深度网络，但它同样可以表示“浅”模型，如分类的逻辑回归。 我们将对我们将生成的合成数据进行简单的逻辑回归，并保存到HDF5格式的数据集中，用以输入到Caffe中。 一旦该模型完成，我们将添加层以提高准确性。 <br />
这就是Caffe的意义：定义一个模型，实验，然后部署。<br />
>
Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative features and 2 noise features.<br />
创建一个10000*4的数据，用于二元分类，具有2个信息特征和2个噪声特征。

**如果在运行过程中，出现“module compiled against API version 0xc but this version of numpy is 0xb”类似的错误**<br />
解决办法：**升级numpy（注意我们这里使用的numpy版本是numpy+mkl版）到最新版**<br />

```
沙@DESKTOP-RMK94TQ /cygdrive/c/caffe/caffe
$ ./scripts/build/tools/Release/caffe train --solver ./examples/hdf5_classificat                                                                                                                                                                                               ion/logreg_solver.prototxt
I0820 17:58:59.517169 51668 caffe.cpp:212] Use CPU.
I0820 17:58:59.518167 51668 solver.cpp:44] Initializing solver from parameters:
train_net: "C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_train.p                                                                                                                                                                                               rototxt"
test_net: "C:\\caffe\\caffe\\examples\\hdf5_classification\\logreg_auto_test.pro                                                                                                                                                                                               totxt"
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
I0820 17:58:59.519665 51668 hdf5_data_layer.cpp:80] Loading list of HDF5 filenam                                                                                                                                                                                               es from: C:\caffe\caffe\examples\hdf5_classification\data\train.txt
I0820 17:58:59.519665 51668 hdf5_data_layer.cpp:94] Number of HDF5 files: 2
I0820 17:58:59.521162 51668 hdf5.cpp:32] Datatype class: H5T_FLOAT
I0820 17:58:59.521661 51668 net.cpp:122] Setting up data
I0820 17:58:59.521661 51668 net.cpp:129] Top shape: 10 4 (40)
I0820 17:58:59.521661 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.521661 51668 net.cpp:137] Memory required for data: 200
I0820 17:58:59.521661 51668 layer_factory.cpp:58] Creating layer label_data_1_sp                                                                                                                                                                                               lit
I0820 17:58:59.521661 51668 net.cpp:84] Creating Layer label_data_1_split
I0820 17:58:59.521661 51668 net.cpp:406] label_data_1_split <- label
I0820 17:58:59.522159 51668 net.cpp:380] label_data_1_split -> label_data_1_spli                                                                                                                                                                                               t_0
I0820 17:58:59.522159 51668 net.cpp:380] label_data_1_split -> label_data_1_spli                                                                                                                                                                                               t_1
I0820 17:58:59.522159 51668 net.cpp:122] Setting up label_data_1_split
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.522159 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.522159 51668 net.cpp:137] Memory required for data: 280
I0820 17:58:59.522159 51668 layer_factory.cpp:58] Creating layer ip1
I0820 17:58:59.522159 51668 net.cpp:84] Creating Layer ip1
I0820 17:58:59.522159 51668 net.cpp:406] ip1 <- data
I0820 17:58:59.522159 51668 net.cpp:380] ip1 -> ip1
I0820 17:58:59.522159 51668 common.cpp:36] System entropy source not available,                                                                                                                                                                                                using fallback algorithm to generate seed instead.
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
I0820 17:58:59.523157 51668 net.cpp:200] accuracy does not need backward computa                                                                                                                                                                                               tion.
I0820 17:58:59.523157 51668 net.cpp:198] ip1_ip1_0_split needs backward computat                                                                                                                                                                                               ion.
I0820 17:58:59.523157 51668 net.cpp:198] ip1 needs backward computation.
I0820 17:58:59.523157 51668 net.cpp:200] label_data_1_split does not need backwa                                                                                                                                                                                               rd computation.
I0820 17:58:59.523157 51668 net.cpp:200] data does not need backward computation                                                                                                                                                                                               .
I0820 17:58:59.523157 51668 net.cpp:242] This network produces output accuracy
I0820 17:58:59.523157 51668 net.cpp:242] This network produces output loss
I0820 17:58:59.523157 51668 net.cpp:255] Network initialization done.
I0820 17:58:59.523656 51668 solver.cpp:172] Creating test net (#0) specified by                                                                                                                                                                                                test_net file: C:\caffe\caffe\examples\hdf5_classification\logreg_auto_test.prot                                                                                                                                                                                               otxt
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
I0820 17:58:59.523656 51668 hdf5_data_layer.cpp:80] Loading list of HDF5 filenam                                                                                                                                                                                               es from: C:\caffe\caffe\examples\hdf5_classification\data\test.txt
I0820 17:58:59.524155 51668 hdf5_data_layer.cpp:94] Number of HDF5 files: 1
I0820 17:58:59.525652 51668 net.cpp:122] Setting up data
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 4 (40)
I0820 17:58:59.525652 51668 net.cpp:129] Top shape: 10 (10)
I0820 17:58:59.525652 51668 net.cpp:137] Memory required for data: 200
I0820 17:58:59.525652 51668 layer_factory.cpp:58] Creating layer label_data_1_sp                                                                                                                                                                                               lit
I0820 17:58:59.525652 51668 net.cpp:84] Creating Layer label_data_1_split
I0820 17:58:59.525652 51668 net.cpp:406] label_data_1_split <- label
I0820 17:58:59.525652 51668 net.cpp:380] label_data_1_split -> label_data_1_spli                                                                                                                                                                                               t_0
I0820 17:58:59.525652 51668 net.cpp:380] label_data_1_split -> label_data_1_spli                                                                                                                                                                                               t_1
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
I0820 17:58:59.526151 51668 net.cpp:200] accuracy does not need backward computa                                                                                                                                                                                               tion.
I0820 17:58:59.526151 51668 net.cpp:198] ip1_ip1_0_split needs backward computat                                                                                                                                                                                               ion.
I0820 17:58:59.526151 51668 net.cpp:198] ip1 needs backward computation.
I0820 17:58:59.526151 51668 net.cpp:200] label_data_1_split does not need backwa                                                                                                                                                                                               rd computation.
I0820 17:58:59.526151 51668 net.cpp:200] data does not need backward computation                                                                                                                                                                                               .
I0820 17:58:59.526151 51668 net.cpp:242] This network produces output accuracy
I0820 17:58:59.526151 51668 net.cpp:242] This network produces output loss
I0820 17:58:59.526151 51668 net.cpp:255] Network initialization done.
I0820 17:58:59.526650 51668 solver.cpp:56] Solver scaffolding done.
I0820 17:58:59.526650 51668 caffe.cpp:249] Starting Optimization
I0820 17:58:59.526650 51668 solver.cpp:272] Solving
I0820 17:58:59.526650 51668 solver.cpp:273] Learning Rate Policy: step
I0820 17:58:59.526650 51668 solver.cpp:330] Iteration 0, Testing net (#0)
I0820 17:58:59.530144 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .5124
I0820 17:58:59.530144 51668 solver.cpp:397]     Test net output #1: loss = 0.808                                                                                                                                                                                               089 (* 1 = 0.808089 loss)
I0820 17:58:59.530144 51668 solver.cpp:218] Iteration 0 (0 iter/s, 0.003s/1000 i                                                                                                                                                                                               ters), loss = 0.87678
I0820 17:58:59.530144 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.5
I0820 17:58:59.530144 51668 solver.cpp:237]     Train net output #1: loss = 0.87                                                                                                                                                                                               678 (* 1 = 0.87678 loss)
I0820 17:58:59.530144 51668 sgd_solver.cpp:105] Iteration 0, lr = 0.01
I0820 17:58:59.541124 51668 solver.cpp:330] Iteration 1000, Testing net (#0)
I0820 17:58:59.543619 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7644
I0820 17:58:59.543619 51668 solver.cpp:397]     Test net output #1: loss = 0.591                                                                                                                                                                                               92 (* 1 = 0.59192 loss)
I0820 17:58:59.543619 51668 solver.cpp:218] Iteration 1000 (76923.1 iter/s, 0.01                                                                                                                                                                                               3s/1000 iters), loss = 0.547078
I0820 17:58:59.543619 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.543619 51668 solver.cpp:237]     Train net output #1: loss = 0.54                                                                                                                                                                                               7078 (* 1 = 0.547078 loss)
I0820 17:58:59.543619 51668 sgd_solver.cpp:105] Iteration 1000, lr = 0.01
I0820 17:58:59.556095 51668 solver.cpp:330] Iteration 2000, Testing net (#0)
I0820 17:58:59.558092 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7532
I0820 17:58:59.558092 51668 solver.cpp:397]     Test net output #1: loss = 0.587                                                                                                                                                                                               112 (* 1 = 0.587112 loss)
I0820 17:58:59.558092 51668 solver.cpp:218] Iteration 2000 (71428.6 iter/s, 0.01                                                                                                                                                                                               4s/1000 iters), loss = 0.644126
I0820 17:58:59.558092 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.558092 51668 solver.cpp:237]     Train net output #1: loss = 0.64                                                                                                                                                                                               4126 (* 1 = 0.644126 loss)
I0820 17:58:59.558092 51668 sgd_solver.cpp:105] Iteration 2000, lr = 0.01
I0820 17:58:59.570567 51668 solver.cpp:330] Iteration 3000, Testing net (#0)
I0820 17:58:59.573063 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7572
I0820 17:58:59.573063 51668 solver.cpp:397]     Test net output #1: loss = 0.593                                                                                                                                                                                               993 (* 1 = 0.593993 loss)
I0820 17:58:59.573063 51668 solver.cpp:218] Iteration 3000 (71428.6 iter/s, 0.01                                                                                                                                                                                               4s/1000 iters), loss = 0.57215
I0820 17:58:59.573063 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.573063 51668 solver.cpp:237]     Train net output #1: loss = 0.57                                                                                                                                                                                               215 (* 1 = 0.57215 loss)
I0820 17:58:59.573063 51668 sgd_solver.cpp:105] Iteration 3000, lr = 0.01
I0820 17:58:59.585541 51668 solver.cpp:330] Iteration 4000, Testing net (#0)
I0820 17:58:59.587536 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7644
I0820 17:58:59.587536 51668 solver.cpp:397]     Test net output #1: loss = 0.591                                                                                                                                                                                               92 (* 1 = 0.59192 loss)
I0820 17:58:59.587536 51668 solver.cpp:218] Iteration 4000 (71428.6 iter/s, 0.01                                                                                                                                                                                               4s/1000 iters), loss = 0.547078
I0820 17:58:59.587536 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.587536 51668 solver.cpp:237]     Train net output #1: loss = 0.54                                                                                                                                                                                               7078 (* 1 = 0.547078 loss)
I0820 17:58:59.587536 51668 sgd_solver.cpp:105] Iteration 4000, lr = 0.01
I0820 17:58:59.599511 51668 solver.cpp:330] Iteration 5000, Testing net (#0)
I0820 17:58:59.601508 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7532
I0820 17:58:59.601508 51668 solver.cpp:397]     Test net output #1: loss = 0.587                                                                                                                                                                                               112 (* 1 = 0.587112 loss)
I0820 17:58:59.601508 51668 solver.cpp:218] Iteration 5000 (76923.1 iter/s, 0.01                                                                                                                                                                                               3s/1000 iters), loss = 0.644126
I0820 17:58:59.601508 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.601508 51668 solver.cpp:237]     Train net output #1: loss = 0.64                                                                                                                                                                                               4126 (* 1 = 0.644126 loss)
I0820 17:58:59.602012 51668 sgd_solver.cpp:105] Iteration 5000, lr = 0.001
I0820 17:58:59.614982 51668 solver.cpp:330] Iteration 6000, Testing net (#0)
I0820 17:58:59.616979 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7776
I0820 17:58:59.616979 51668 solver.cpp:397]     Test net output #1: loss = 0.588                                                                                                                                                                                               383 (* 1 = 0.588383 loss)
I0820 17:58:59.616979 51668 solver.cpp:218] Iteration 6000 (66666.7 iter/s, 0.01                                                                                                                                                                                               5s/1000 iters), loss = 0.583577
I0820 17:58:59.616979 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.616979 51668 solver.cpp:237]     Train net output #1: loss = 0.58                                                                                                                                                                                               3577 (* 1 = 0.583577 loss)
I0820 17:58:59.616979 51668 sgd_solver.cpp:105] Iteration 6000, lr = 0.001
I0820 17:58:59.628458 51668 solver.cpp:330] Iteration 7000, Testing net (#0)
I0820 17:58:59.630455 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7792
I0820 17:58:59.630954 51668 solver.cpp:397]     Test net output #1: loss = 0.588                                                                                                                                                                                               225 (* 1 = 0.588225 loss)
I0820 17:58:59.630954 51668 solver.cpp:218] Iteration 7000 (76923.1 iter/s, 0.01                                                                                                                                                                                               3s/1000 iters), loss = 0.5487
I0820 17:58:59.630954 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.630954 51668 solver.cpp:237]     Train net output #1: loss = 0.54                                                                                                                                                                                               87 (* 1 = 0.5487 loss)
I0820 17:58:59.630954 51668 sgd_solver.cpp:105] Iteration 7000, lr = 0.001
I0820 17:58:59.641932 51668 solver.cpp:330] Iteration 8000, Testing net (#0)
I0820 17:58:59.644428 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7772
I0820 17:58:59.644428 51668 solver.cpp:397]     Test net output #1: loss = 0.587                                                                                                                                                                                               423 (* 1 = 0.587423 loss)
I0820 17:58:59.644428 51668 solver.cpp:218] Iteration 8000 (76923.1 iter/s, 0.01                                                                                                                                                                                               3s/1000 iters), loss = 0.670083
I0820 17:58:59.644925 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.644925 51668 solver.cpp:237]     Train net output #1: loss = 0.67                                                                                                                                                                                               0083 (* 1 = 0.670083 loss)
I0820 17:58:59.644925 51668 sgd_solver.cpp:105] Iteration 8000, lr = 0.001
I0820 17:58:59.656404 51668 solver.cpp:330] Iteration 9000, Testing net (#0)
I0820 17:58:59.659397 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .7784
I0820 17:58:59.659397 51668 solver.cpp:397]     Test net output #1: loss = 0.588                                                                                                                                                                                               682 (* 1 = 0.588682 loss)
I0820 17:58:59.659397 51668 solver.cpp:218] Iteration 9000 (71428.6 iter/s, 0.01                                                                                                                                                                                               4s/1000 iters), loss = 0.580133
I0820 17:58:59.659397 51668 solver.cpp:237]     Train net output #0: accuracy =                                                                                                                                                                                                0.7
I0820 17:58:59.659397 51668 solver.cpp:237]     Train net output #1: loss = 0.58                                                                                                                                                                                               0133 (* 1 = 0.580133 loss)
I0820 17:58:59.659898 51668 sgd_solver.cpp:105] Iteration 9000, lr = 0.001
I0820 17:58:59.670877 51668 solver.cpp:447] Snapshotting to binary proto file C:                                                                                                                                                                                               \caffe\caffe\examples\hdf5_classification\data\train_iter_10000.caffemodel
I0820 17:58:59.671376 51668 sgd_solver.cpp:273] Snapshotting solver state to bin                                                                                                                                                                                               ary proto file C:\caffe\caffe\examples\hdf5_classification\data\train_iter_10000                                                                                                                                                                                               .solverstate
I0820 17:58:59.671875 51668 solver.cpp:310] Iteration 10000, loss = 0.548088
I0820 17:58:59.671875 51668 solver.cpp:330] Iteration 10000, Testing net (#0)
I0820 17:58:59.673871 51668 solver.cpp:397]     Test net output #0: accuracy = 0                                                                                                                                                                                               .78
I0820 17:58:59.673871 51668 solver.cpp:397]     Test net output #1: loss = 0.588                                                                                                                                                                                               317 (* 1 = 0.588317 loss)
I0820 17:58:59.673871 51668 solver.cpp:315] Optimization Done.
I0820 17:58:59.673871 51668 caffe.cpp:260] Optimization Done.
```
