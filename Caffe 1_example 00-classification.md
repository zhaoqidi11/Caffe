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
#关于平均的说明，见	https://blog.csdn.net/yangdashi888/article/details/79340195
print mu


mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
#首先在计算机中，该图像以(R,G,B)格式来存储，先对G压缩，得到(R,B)的平均值；再对B压缩，得到整体所有的像素的平均值(R)
print 'mean-subtracted values:', zip('BGR', mu)

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

############CPU分类过程#####################

#设定输入图像大小
net.blobs['data'].reshape(50,#batch大小，在这次测试中，载入了五十张相同的图（就这个cat.jpg)
                          3,#3-channel（BGR）images
                          227,227)#图像大小为227x227
#使用load_image函数载入图像
image = caffe.io.load_image(caffe_root + '\\caffe\\examples\\images\cat.jpg')
#将image读入到'data'中，并对数据进行处理（如43行到62行写的那样）
transformed_image = transformer.preprocess('data', image)

################显示图片
plt.imshow(image)
plt.show()

#################################
#将转换后的图像载入到blobs中以'data'参数存入,后面data[...]猜测表示传入到所有维度中
net.blobs['data'].data[...] = transformed_image
#网络前馈输出到output中，output是一个dict类型的数据（字典）
output = net.forward()
#取出来'prob'，也就是第一张图（其实这五十张都是一样的）的概率
output_prob = output['prob'][0]

#对这1000个概率进行排序，找出来最大的那一个，那个类别就是我们预测的类别（类似最大似然估计
print 'predicted class is:', output_prob.argmax()#找到概率最大值的索引

#载入文件
labels_file = caffe_root + '\\caffe\\data\\ilsvrc12\\synset_words.txt'
#我们读入这个文件
labels = np.loadtxt(labels_file, str, delimiter = '\t')#从txt文件中读取到向量中，str代表该文本文件含字母，delimiter代表分隔符

print 'output label: ', labels[output_prob.argmax()]#print 后加逗号的作用是-------输出在同一行。

#对概率进行排序（从低到高），取倒数5个（即前五个最高的）
top_inds = output_prob.argsort()[::-1][:5]
#Python序列切片地址可以写为[开始：结束：步长]，其中的开始和结束可以省略

#打印概率的前五名以及相应的标签
print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])

#net.blobs是一个Orderedict，已经按照进入字典的顺序排好序了，我们使用key,value的方式取出它的值
#取出数据的shape

#为什么使用iteritems()的原因：节省内存，见https://blog.csdn.net/TNTIN/article/details/79922577
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


#net.params是卷积神经网络的参数，param[0]代表weights,param[1]代表biases
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

############################可视化特征(卷积核）
def vis_squre(data):

    #正则化数据
    data = (data - data.min()) / (data.max() - data.min())

    #先开方再向正无穷方向取整
    n = int(np.ceil(np.sqrt(data.shape[0])))
    
    #在相邻的卷积核之间加入空白，padding用于填充，padding = (((0,10^2-卷积核总数），(0,1),(0,1))+((0,0),)*(4 - 3) = (0, 4)
    #注意tuple是可以拼接的，tuple操作规则见https://www.cnblogs.com/beiguoxia113/p/6132320.html
    #因此我们可以得到padding = ((0,4),(0,1),(0,1),(0,0))
    #填充方法见https://blog.csdn.net/tan_handsome/article/details/80296827
    #第一维(卷积核个数)前面填充0个填充值后面填充1个填充值；第二维和第三维分别是长和宽，前面都不填充，后面填充1个填充值；第四维(通道数)不填充
    padding = (((0, n ** 2 - data.shape[0]),(0,1),(0,1)) + ((0,0),) * (data.ndim - 3))
    
    #填充方法见https://blog.csdn.net/qq_36332685/article/details/78803622，其中由于我们选择了constant，则固定填充1值，而1值经过转换器之后，即是
    #255
    #填充前，data的shape是(96,11,11,3)，填充后变成(100,12,12,3)
    data = np.pad(data, padding, mode = 'constant', constant_values=1)
    #连接元组(n,n,12,12,3),并交换维度(0,2,1,3,5)即变成(n,12,n,12,3)，即(10,12,10,12,3)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    #再次转换维度变成(120,10,3)
    data = data.reshape((n * data.shape[1], n * data.shape[3]) +data.shape[4:])
    #展示图像
    plt.imshow(data); plt.axis('off');plt.show()

#展示第一个卷积核的图像（weight）
filters = net.params['conv1'][0].data
#将卷积核卷积核的维度进行转置，原来是(96,3,11,11)，现在转变成(96,11,11,3)，96为卷积核的数量，11 x 11代表卷积核的大小，3代表通道的数量
vis_square(filters.transpose(0,2,3,1))

#展示经过第一个卷积核之后的图像，0代表第一个维度仅取第0个下标的数据（第一张图像），第二维取前36个数据（即第0到第35个数据，前36个数据）
#关于[0, :36]的用法，见https://blog.csdn.net/csj664103736/article/details/72828584
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)

#展示经过第5个池化层之后的图像，仅取第一张图像
feat = net.blobs['pool5'].data[0]
vis_square(feat)

###########绘制全连接层(fc6)的输出
feat = net.blobs['fc6'].data[0]
#subplot，产生子图，具体用法见https://www.jianshu.com/p/de223a79217a
#前两个数据(2,1)代表产生2行1列的图像，第三个数据1代表是这是第一张图（序号1）
plt.subplot(2, 1, 1)

plt.plot(feat.flat)
#第三个数据2代表这是第二张图
plt.subplot(2, 1, 2)
#plt.hist代表绘制
a = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()







```
