Net Surgery
----
一些可能有用的资料：<br />
caffe之python接口实战 ：net_surgery 官方教程源码解析https://blog.csdn.net/xuluohongshang/article/details/78700350 <br />
net_surgery中如何将全连接层转换成卷积层:https://blog.csdn.net/weixin_38208741/article/details/78661533 <br />
caffe学习笔记（8）：Net Surgery: https://blog.csdn.net/qq_30401249/article/details/51472688?locationNum=3&fps=1 <br />
caffe学习笔记11 -- Net Surgery : https://blog.csdn.net/thystar/article/details/50681609 <br />

通过编辑模型参数，可以将Caffe网络转换为特定需求。 网络的数据(data)，差异(diffs)和参数(parameters)都在pycaffe中暴露。
>
#### 设计Filters
为了展示如何加载，操作和保存参数，我们将自己的Filters设计到一个只有单个卷积层的简单网络中。<br />
该网络有两个blob，用于输入的数据(data)和用于卷积输出的conv和用于卷积滤波器权重和偏差的一个参数conv。
```
import numpy as np
import matplotlib.pyplot as plt

caffe_root = 'C:\\caffe'
import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')

import caffe
import os

os.chdir(caffe_root+'\\caffe\\examples')

# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net('net_surgery/conv.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

# load image and prepare as a single input batch for Caffe
im = np.array(caffe.io.load_image('images/cat_gray.jpg', color=False)).squeeze()#squeeze()除去size为1的维度
plt.title("original image")
plt.imshow(im)
plt.axis('off')
plt.show()

im_input = im[np.newaxis, np.newaxis, :, :]
#关于newaxis的介绍：https://blog.csdn.net/zjm750617105/article/details/53376257，https://blog.csdn.net/lanchunhui/article/details/49725065
#最初，星号变量是用在函数的参数传递上的，在下面的实例中，单个星号代表这个位置接收任意多个非关键字参数，在函数的*b位置上将其转化成元组（tuple）
#关于星号的用法：https://www.cnblogs.com/empty16/p/6229538.html
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

# helper show filter outputs
def show_filters(net):
    net.forward()
    plt.figure()#net.blobs['conv'].data代表的是卷积之后的图像，shape大小是(1,3,356,476)3是卷积核个数，后面两维是图像大小
    filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()#保存了图像颜色最浅和最深的参数
    for i in range(3):
        plt.subplot(1,4,i+2)#1行4列，这是第i+2个图像（序号从1开始）
        plt.title("filter #{} output".format(i))#net.blobs['conv'].data[0,i]第一个维度是图片个数，第二个维度是第几个核，依次输出3个核的卷积结果

        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)#imshow将数据标准化为最小和最大值。 您可以使用vmin和vmax参数或norm参数来控制（如果您想要非线性缩放）。
        plt.tight_layout()#tight_layout会自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分。
        #关于tight_layout的介绍https://blog.csdn.net/wizardforcel/article/details/54233181
        plt.axis('off')
    plt.show()


# filter the image with initial 
show_filters(net)

# pick first filter output
conv0 = net.blobs['conv'].data[0, 0]#net.blobs['conv'].data的shape是1 x 3 x 356 x 476
print("pre-surgery output mean {:.2f}".format(conv0.mean()))
# set first filter bias to 1
net.params['conv'][1].data[0] = 1.#卷积层有两个变量（weight['conv'][0]和偏置['conv'][1]）我们令第一个偏置(data[0])=1（初始化设置的是constant，数值为0）
net.forward()
print("post-surgery output mean {:.2f}".format(conv0.mean()))

'''
改变滤波器权重更令人兴奋，因为我们可以分配任何内核，如高斯模糊，边缘的Sobel算子等等。 以下手术将第0个滤镜转换为高斯模糊，将第1和第2滤镜转换为Sobel算子的水平和垂直渐变部分。
看第0个输出是如何模糊的，第1个选择水平边缘，第2个选择垂直边缘。
'''

ksize = net.params['conv'][0].data.shape[2:]#net.params['conv'][0].data.shape is 3 x 1 x 5 x 5
# make Gaussian blur
sigma = 1.
#关于mgrid的介绍：https://blog.csdn.net/Wzz_Liu/article/details/80962403
#简单来说mgrid就是，第i个元素的第i维的数字由mgrid函数接受的第i个参数决定，其他维由其他维相应的数字决定，但是以广播填充的形式
y, x = np.mgrid[-ksize[0]//2 + 1:ksize[0]//2 + 1, -ksize[1]//2 + 1:ksize[1]//2 + 1]#双斜杠（//）表示地板除，即先做除法（/），然后向下取整（floor）
g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))#2维高斯函数的公式，进行高斯初始化
#python中，双星号x**2表示x^2，来源：https://blog.csdn.net/fengjiexyb/article/details/77460510
gaussian = (g / g.sum()).astype(np.float32)#astype：转换类型，转换成32位浮点数
net.params['conv'][0].data[0] = gaussian#将W初始化为gaussian (一共3个核，只初始化第一个，后2个不进行初始化
# make Sobel operator for edge detection
net.params['conv'][0].data[1:] = 0.#net.params['conv'][0].data.shape = 3 x 1 x 5 x 5 第一个维度代表是第几个核
sobel = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1), dtype=np.float32).reshape((3,3))
net.params['conv'][0].data[1, 0, 1:-1, 1:-1] = sobel  # horizontal -1表示导数第二个数
net.params['conv'][0].data[2, 0, 1:-1, 1:-1] = sobel.T  # vertical
show_filters(net)

#通过net surgery，参数可以通过网络移植，通过自定义的每个参数操作进行规范化，并根据您的方案进行转换。
```
现在将caffe中自带的ImageNet模型“caffenet”转换成一个全卷积网络，以便于对大量输入高效，密集的运算。这个模型产生一个与输入相同大小的分类图而不是单一的分类器。特别的，在451x451的输入中，一个8x8的分类图提供了64倍的输出但是仅消耗了3倍的时间。The computation exploits a natural efficiency of convolutional network (convnet) structure by amortizing the computation of overlapping receptive fields.

我们将CaffeNet的InnerProduct矩阵乘法层转换为卷积层，这是仅有的改变: the other layer types are agnostic to spatial size. 卷积是平移不变的, 激活是元素操作。 fc6全连接层用fc6-conv替换，用6*6滤波器已间隔为1对pool5的输出进行滤波。回到图像空间中，对每个227*227的输入且步长为32的图像给定一个分类器, 输出图像和感受野的尺寸相同；　output = (input -kernel_size) / stride + 1。
### Linux diff命令用法：
diff命令在最简单的情况下，比较给定的两个文件的不同。如果使用“-”代替“文件”参数，则要比较的内容将来自标准输入。diff命令是以逐行的方式，比较文本文件的异同处。如果该命令指定进行目录的比较，则将会比较该目录中具有相同文件名的文件，而不会对其子目录文件进行任何比较操作。<br />
介绍来自：http://man.linuxde.net/diff
比较bvlc_caffenet_full_conv与bvlc_reference_caffenet文件夹下的deploy.prototxt的区别<br />
输出结果如下<br />
```
 diff ./examples/net_surgery/bvlc_caffenet_full_conv.prototxt ./models/bvlc_reference_caffenet/deploy.prototxt
1,2c1
< # Fully convolutional network version of CaffeNet.
< name: "CaffeNetConv"
---
> name: "CaffeNet"
7,11c6
<   input_param {
<     # initial shape for a fully convolutional network:
<     # the shape can be set for each input by reshape.
<     shape: { dim: 1 dim: 3 dim: 451 dim: 451 }
<   }
---
>   input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }
157,158c152,153
<   name: "fc6-conv"
<   type: "Convolution"
---
>   name: "fc6"
>   type: "InnerProduct"
160,161c155,156
<   top: "fc6-conv"
<   convolution_param {
---
>   top: "fc6"
>   inner_product_param {
163d157
<     kernel_size: 6
169,170c163,164
<   bottom: "fc6-conv"
<   top: "fc6-conv"
---
>   bottom: "fc6"
>   top: "fc6"
175,176c169,170
<   bottom: "fc6-conv"
<   top: "fc6-conv"
---
>   bottom: "fc6"
>   top: "fc6"
182,186c176,180
<   name: "fc7-conv"
<   type: "Convolution"
<   bottom: "fc6-conv"
<   top: "fc7-conv"
<   convolution_param {
---
>   name: "fc7"
>   type: "InnerProduct"
>   bottom: "fc6"
>   top: "fc7"
>   inner_product_param {
188d181
<     kernel_size: 1
194,195c187,188
<   bottom: "fc7-conv"
<   top: "fc7-conv"
---
>   bottom: "fc7"
>   top: "fc7"
200,201c193,194
<   bottom: "fc7-conv"
<   top: "fc7-conv"
---
>   bottom: "fc7"
>   top: "fc7"
207,211c200,204
<   name: "fc8-conv"
<   type: "Convolution"
<   bottom: "fc7-conv"
<   top: "fc8-conv"
<   convolution_param {
---
>   name: "fc8"
>   type: "InnerProduct"
>   bottom: "fc7"
>   top: "fc8"
>   inner_product_param {
213d205
<     kernel_size: 1
219c211
<   bottom: "fc8-conv"
---
>   bottom: "fc8"

```
可见，结构上唯一需要改变的就是将全连接的分类器内积层改为卷积层，并且使用6\*6的滤波器，因为参考模型的分类以pool6的36个输出作为fc6-conv的输入。为了保证密集分类，令步长为1。注意，重命名是为了避免当模型命名为"预训练"模型时caffe取载入旧的参数。 

全连接层如何转换成卷积层：<br />
fc6的params的shape是：4096 x 9216 <br />
fc6-conv的params的shape是：4096 x 256 x 6 x 6<br />
fc7的params的shape是：4096 x 4096 <br />
fc7-conv的params的shape是：4096 x 4096 x 1 x 1<br />
fc8的params的shape是：1000 x 4096 <br />
fc8-conv的shape是：1000 x 4096 x 1 x 1<br />


卷积权重由 output\*input\*heigth\*width
的规模决定，为了将内部产生的权重对应到卷积滤波器中，需要将内部产生的权值转变为channel\*height\*width规模的滤波矩阵。但是他们全部在内存中(按行存储), 所以我们可以直接指定，即两者是一致的。
<br />
偏置与内连接层相同。<br />
