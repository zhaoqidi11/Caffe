Fine-tuning a Pretrained Network for Style Recognition<br />
微调一个预先训练的风格识别网络<br />
预训练Caffe Network并且微调参数<br />

这种方法的优点是，由于预先训练的网络是在大量图像上学习的，中间层捕获了一般视觉外观的“语义”。<br />
可以把它看作一个非常强大的通用视觉特性，您可以将其视为一个黑盒。除此之外，要想在目标任务中<br />
获得良好的性能，只需要相对较少的数据。<br />

需要准备数据，（1）使用提供的Shell脚本获取ImageNet ilsvrc预训练模型；<br />
（2）下载完整的Flickr样式数据集的子集<br />
（3）将下载的Flickr数据集编译到Caffe可以使用的数据库中<br />
>
以下是运行过程：<br />
(1)在cygwin中运行get_ilsvrc_aux.sh<br />
切换到根目录/caffe/data/ilsvrc12中，输入如下命令<br />
```
bash get_ilsvrc_aux.sh
```
(2)在cmd中运行download_model_binary.py<br />
切换到Caffe根目录下，输入如下命令<br />
```
python scripts/download_model_binary.py models/bvlc_reference_caffenet
```
(3)在cmd中下载Flickr的数据<br />
```
python examples/finetune_flickr_style/assemble_data.py ^
    --workers=-1  --seed=1701 ^
    --images=2000  --label=5
```
(原代码如下）<br />
```Python
# Download just a small subset of the data for this exercise.
# (2000 of 80K images, 5 of 20 labels.)
# To download the entire dataset, set `full_dataset = True`.
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5

# This downloads the ilsvrc auxiliary data (mean file, etc),
# and a subset of 2000 images for the style recognition task.
import os
os.chdir(caffe_root)  # run scripts from caffe root
#!data/ilsvrc12/get_ilsvrc_aux.sh
!scripts/download_model_binary.py models/bvlc_reference_caffenet
!python examples/finetune_flickr_style/assemble_data.py \
    --workers=-1  --seed=1701 \
    --images=$NUM_STYLE_IMAGES  --label=$NUM_STYLE_LABELS
# back to examples
os.chdir('examples')
```
全部代码如下<br />

```Python
'''
关于这个代码的一些解读：
https://blog.csdn.net/thystar/article/details/51258613?locationNum=5&fps=1
http://www.mamicode.com/info-detail-1307306.html
https://blog.csdn.net/u012614287/article/details/80609350
https://blog.csdn.net/tianliangjay/article/details/51355676
https://blog.csdn.net/sinat_26917383/article/details/54999868
https://blog.csdn.net/ychan_cc/article/details/69569390
https://blog.csdn.net/thystar/article/details/51258613
https://blog.csdn.net/qq_38156052/article/details/80916252
'''




import sys
caffe_root = 'C:\\caffe'
sys.path.insert(0, caffe_root + '\\caffe\\python')
import caffe

#caffe.set_device(0)
#caffe.set_mode_gpu()//这是设置gpu模式要写这两条
caffe.set_mode_cpu()

import numpy as np
from pylab import *
import tempfile

### 定义图片格式转换函数
def deprocess_net_image(image):
    """将caffe格式图片转化为python格式"""
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB，Python切片介绍：https://www.cnblogs.com/hiwuchong/p/8052502.html
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)#将数据四舍五入到指定位数
    #参考：https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.round.html
    image = np.require(image, dtype=np.uint8)#返回满足要求的ndarray

    return image


####################################################
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5



################读取权重
import os
weights = os.path.join(caffe_root, 'caffe\\models\\bvlc_reference_caffenet\\bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)#assert的用法https://www.cnblogs.com/hezhiyao/p/7805278.html


################读取标签（原来的ImageNet标签）
# Load ImageNet labels to imagenet_labels
imagenet_label_file = caffe_root + '\\caffe\\data\\ilsvrc12\\synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
assert len(imagenet_labels) == 1000 #Python join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

#join函数的用法:http://www.runoob.com/python/att-string-join.html
#str = "-";
#seq = ("a", "b", "c"); # 字符串序列
#print str.join( seq );
#输出结果：a-b-c

print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])#s输出前十个标签

# Load style labels to style_labels
style_label_file = caffe_root + '\\caffe\\examples\\finetune_flickr_style\\style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]#只保留钱5个标签
print '\nLoaded style labels:\n', ', '.join(style_labels)


###定义和跑网络
###我们开始定义caffenet，这一个函数，定义了CaffeNet结构（AlexNet上的一个小变体），并接受指定数据和输出类数量的参数

from caffe import layers as L
from caffe import params as P


# 定义权值学习策略
# 神经网络逐层的学习参数，本例中除最后一层外所有层均采用这些参数
weight_param = dict(lr_mult=1, decay_mult=1)#字典的赋值方式
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2
# 将学习率因子置为0,来冻结层参数

# bottom：传入数据
# nout：输出核数目
# param：决定是否需要更新权值
###卷积层+整流层
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    """定义一个网络,便于赋予一些参数初始值和直接调用"""
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


###全连接层+ReLU
def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

###池化层
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """　便于定义新的数据格式和参数选择, 用来生成deploy.prototxt文件"""

    # 返回指定CaffeNet的NetSpec，该规范遵循原始的[proto text]规范（specification)
    # 以下说明来自https://blog.csdn.net/sinat_26917383/article/details/54999868
    # data代表训练数据
    # label代表新数据标签，跟num_classes类似，跟给出数据的标签数量一致，譬如新数据的标签有5类，那么就是5
    # train,是否开始训练，默认为true
    # num_classes代表fine-tuning原来的模型的标签数，如果fine-tuning是ImageNet是千分类，那么num_classes=1000
    # classifier_name，最后的全连接层名字，如果是fine-tuning需要重新训练的话，则需要修改最后的全连接层
    # learn_all，这个变量用于将学习率设置为0，在caffenet中，如果learn_all=False,则使用frozen_param设置网络层的学习率，即学习率为0 
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""

    #label与num_classes:label代表这次训练要使用的标签的数量，num_classess代表要微调的模型原来的标签数量
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:#训练时采用dropout
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)#由于没有指定初始化方式，在这里默认为所有权重都是0，InnerProduct层解析https://blog.csdn.net/bailufeiyan/article/details/50864675
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    #setattr用法：http://www.runoob.com/python/python-func-setattr.html
        #用法：def __setattr__(self, name, value):
       # self.tops[name] = value

    if not train:#测试时，增加softmax输出层
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # 将网络写入临时文件并返回其文件名
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

# 现在，让我们创建一个CaffeNet，它以未标记的“dummy data”作为输入，
# 允许我们在外部设置它的输入图片，并查看它所预测的ImageNet类。
dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
# 一个dummydata层产生随机数据(num,channel,height,width)这里输出全为０,用于调试
imagenet_net_filename = caffenet(data=dummy_data, train=False)
# 产生image_net.prototxt文件
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)
# 初始化/装载一个预训练好的网络：deploy.prototxt文件, 权值文件,　开始盖楼测试输出网络参数



'''
    定义一个函数style_net调用caffenet,输入数据参数为Flicker style数据集
    网络的输入是下载的Flickr style数据集，层类型为ImageData
    输出是20个类别
    分类层重命名为fc8_flickr以告诉caffe不要加载原始分类的那一层的权值。
'''

    # train:训练时用于mirror
    # subset:决定调用训练数据还是测试数据
def style_net(train=True, learn_all=False, subset=None):
    """调用新的数据层,导入数据,调用caffeNet生成flickr_style的deploy.prototxt文档"""
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + '\\caffe\\data\\flickr_style\\%s.txt' % subset
        #图片从txt里面的图片路径导入; source: each line give an images filename and label
    #占位符用法https://blog.csdn.net/u013216667/article/details/51316971 
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + '\\caffe\\data\\ilsvrc12\\imagenet_mean.binaryproto')
################################################################################
# data,label:   为top的名称
# ImageData():  表示该层类型为数据层，数据来自于图片。
# source:       一个文本文件的名称，每一行给定一个图片文件的名称和标签。
# batch_size:   每一次处理数据的个数。
# new_width:    图片resize的宽。（可选）
# new——height:  图片resize的高。（可选）
# ntop:         表明有多少个blobs数据输出，示例中为2，代表着data和label。
# transform_param:  数据预处理
#   crop_size:  对图像进行裁剪。如果定义了crop_size，那么在train时会对大
#               于crop_size的图片进行随机裁剪，而在test时只是截取中间部分。
#   mean_value: 图像通道的均值。三个值表示RGB图像中三个通道的均值。
#   mirror:     图像镜像。True为使用镜像。
################################################################################
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)

#•用上面定义的style_net函数来初始化untrained_style_net：下载的数据集作为图像输入，加载预训练模型为权重。

#•调用untrained_style_net.forward()得到一个batch的训练数据

untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
                                weights, caffe.TEST)
untrained_style_net.forward()
style_data_batch = untrained_style_net.blobs['data'].data.copy()
style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)
#关于dtype的介绍：https://www.jianshu.com/p/621af296dcd5
'''
从第一批训练数据的50张图像中选择一张（这里选择第8张）显示该图片，然后在imagenet_net上运行，
ImageNet预训练网络给出前5个得分最高的类别
下面我们选择一张网络可以给出合理预测的图片，因为该图像是一张海滩图片，而“沙洲”和“海岸”
这两个类别是存在于ImageNet的1000个类别中的，对于其他的图片，预测结果未必好，有些是由于网络
识别物体错误，但是跟有可能是由于ImageNet的1000个类别中没有被识别图片的类别。修改batch_index
的值将默认的9改为0-49中的任意一个数字，观察其他图像的预测结果。



'''


def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    #enumerate()用法见http://www.runoob.com/python/python-func-enumerate.html

    #%5.2f的意思是，总输出为5位，其中小数2为，后面的%%代表百分号


#将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')


batch_index = 8
image = style_data_batch[batch_index]
plt.imshow(deprocess_net_image(image))
show()
print 'actual label =', style_labels[style_label_batch[batch_index]]

disp_imagenet_preds(imagenet_net, image)

'''
我们还可以看看untrained_style_net的预测，但是我们不会看到任何有趣的东西，
因为它的分类器还没有经过训练。
事实上，由于我们对分类器进行了零初始化(参见caffenet定义—没有将weight_filler传递到最终的内积层--fc8_flickr)，
softmax输入应该为零，因此我们应该看到每个标签(对于N个标签)的预测概率为1/N。因为我们设置了N = 5，我们得到了每
个类20%的预测概率。


'''


disp_style_preds(untrained_style_net, image)

'''
top 5 predicted style labels =
         (1) 20.00% Detailed
         (2) 20.00% Pastel
         (3) 20.00% Melancholy
         (4) 20.00% Noir
         (5) 20.00% HDR

可以验证在分类层之前的fc7的激励与ImageNet的预训练模型相同（或者十分相似），因为两个模型在conv1到fc7中使用相同的预训练权值
'''

diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
error = (diff ** 2).sum()
assert error < 1e-8

#删除untrained_style_net以节省内存。(保留imagenet_net，稍后我们将再次使用它。)
del untrained_style_net

'''
定义solver函数创建caffe的solver文件中的参数，用于训练网络（学习权值），
在这个函数中，我们会设置各种参数的值用于learning，display，和“snapshotting”。这些参数的
意思在我之前的博客里都有解释，你可以修改这些值改善预测结果

'''

from caffe.proto import caffe_pb2


#####关于solver的解释：https://blog.csdn.net/sweet0heart/article/details/53042390
def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    #指定训练和测试的网络的位置
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.每1000次训练测试一次
        s.test_iter.append(100) # Test on 100 batches each time we test.每次测试100个batches
    '''
    iter_size*batch size=实际使用的batch size。 
    相当于读取batchsize*itersize个图像才做一下gradient decent。 
    这个参数可以规避由于gpu不足而导致的batchsize的限制 
    因为你可以用多个iteration做到很大的batch 即使单次batch有限

    '''
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    #最大迭代次数
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    #预防过拟合
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    #每1000轮输出损失
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + '\\caffe\\models\\finetune_flickr_style\\finetune_flickr_style'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

'''
现在我们调用求解器来训练style net的分类器

•定义run_solvers函数，输入一个求解器列表（有两个或多个不同的求解器），
它以轮询的方式逐个执行并记录精度和损失，最后保存权重。函数返回 精度、损失、权重。
'''

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    '''
    为niter轮运行solvers
    为niter次迭代运行solvers，返回每个迭代记录的损失和精度。
    ' solvers '是一个(名称，solver)元组的列表。
    '''
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %#%2d表示输出2位整型
                                  (n, loss[n][it], np.round(100*acc[n][it]))#np.round是对括号里面的数进行四舍五入
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    '''os.path.join
    tempfile.mkdtemp（后缀=无，前缀=无，dir =无）
    尽可能以最安全的方式创建临时目录。目录的创建中没有竞争条件。该目录只能通过创建用户ID进行读取，写入和搜索。
    用户mkdtemp()负责在完成后删除临时目录及其内容。
    该前缀，后缀和DIR参数是一样的 mkstemp()。
    mkdtemp() 返回新目录的绝对路径名。

    '''
    #https://blog.csdn.net/baidu_39416074/article/details/80937826
    weights = {}
    weight_dir2 = 'c:\\caffe'
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir2, filename)#路径连接函数：https://blog.csdn.net/zmdzbzbhss123/article/details/52279008
        s.net.save(weights[name])#在训练过程中保存模型参数，来源[Caffe-Python接口常用API参考]：https://blog.csdn.net/langb2014/article/details/53082704
    return loss, acc, weights

'''
•创建两个求解器，一个style_solver通过copy_from的方式加载预训练权重，
另一个scratch_style_solver采用随机初始化网络，将两个求解器作为列表参数调用run_solvers训练网络。
'''

niter = 200  # number of iterations to train

# Reset style_solver as before.
style_solver_filename = solver(style_net(train=True))
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

# For reference, we also create a solver that isn't initialized from
# the pretrained ImageNet weights.
scratch_style_solver_filename = solver(style_net(train=True))
scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', style_solver),
           ('scratch', scratch_style_solver)]
loss, acc, weights = run_solvers(niter, solvers)
print 'Done.'

train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

# Delete solvers to save memory.
del style_solver, scratch_style_solver, solvers

#•绘制上面保存的两种网络的精度值和损失值的对比图

plot(np.vstack([train_loss, scratch_train_loss]).T)#它是垂直（按照行顺序）的把数组给堆叠起来。
xlabel('Iteration #')
ylabel('Loss')
show()

plot(np.vstack([train_acc, scratch_train_acc]).T)
xlabel('Iteration #')
ylabel('Accuracy')
show()

'''
•定义一个评估函数eval_style_net，通过构建一个新的TEST网络来测试上面训练迭代好的权重参数，看一下网络的平均精度。
'''

def eval_style_net(weights, test_iters=10):
    test_net = caffe.Net(style_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy

test_net, accuracy = eval_style_net(style_weights)
print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )
scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights)
print 'Accuracy, trained from   random initialization: %3.1f%%' % (100*scratch_accuracy, )

#4. 端到端微调
#•将style_net的参数设置为learn_all=True来训练所有的层（默认情况是learn_all=False冻结FC1到FC7层）

end_to_end_net = style_net(train=True, learn_all=True)

# Set base_lr to 1e-3, the same as last time when learning only the classifier.
# You may want to play around with different values of this or other
# optimization parameters when fine-tuning.  For example, if learning diverges
# (e.g., the loss gets very large or goes to infinity/NaN), you should try
# decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
# for which learning does not diverge).
base_lr = 0.001

style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(style_weights)

scratch_style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
scratch_style_solver.net.copy_from(scratch_style_weights)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained, end-to-end', style_solver),
           ('scratch, end-to-end', scratch_style_solver)]
_, _, finetuned_weights = run_solvers(niter, solvers)
print 'Done.'

style_weights_ft = finetuned_weights['pretrained, end-to-end']
scratch_style_weights_ft = finetuned_weights['scratch, end-to-end']

# Delete solvers to save memory.
del style_solver, scratch_style_solver, solvers

plt.imshow(deprocess_net_image(image))
disp_style_preds(test_net, image)
show()

batch_index = 1
image = test_net.blobs['data'].data[batch_index]
plt.imshow(deprocess_net_image(image))
show()

print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]


disp_style_preds(test_net, image)

disp_style_preds(scratch_test_net, image)

disp_imagenet_preds(imagenet_net, image)
```
