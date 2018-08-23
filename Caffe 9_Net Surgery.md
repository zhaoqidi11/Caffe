Net Surgery

通过编辑模型参数，可以将Caffe网络转换为特定需求。 网络的数据(data)，差异(diffs)和参数(parameters)都在pycaffe中暴露。
>
#### 设计Filters
为了展示如何加载，操作和保存参数，我们将自己的Filters设计到一个只有单个卷积层的简单网络中。<br />
该网络有两个blob，用于输入的数据(data)和用于卷积输出的conv和用于卷积滤波器权重和偏差的一个参数conv。
