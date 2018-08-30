构建自己的Siamese网络
---
### 0 准备工作

一篇可能有用的问答：
[How to convert an arbitrary dataset to the siamese network dataset format in caffe?](https://stackoverflow.com/questions/41904521/how-to-convert-an-arbitrary-dataset-to-the-siamese-network-dataset-format-in-caf)<br />
>
原文如下：
>You don't need to use the exact same format - this is just a tutorial.... All you need to do is provide one or multiple data layers, with a total of three top Blobs: data, data_p, and sim. You can do that in any way you'd like, e.g. LMDB (like in the MNIST example), HDF5, or whatever.

>**General explanation**<br />
>In the tutorial, they further show and easy way to load the image pairs: you concatenate two images in the channel dimension. For gray-scale, you take two input images, where each has for example the dimension [1, 1, 28, 28] (i.e. 1 image, 1 channel, 28x28 resolution). Then you concatenate them to be one image of size [1, 2, 28, 28] and save them e.g. to an LMDB.<br />
>In the network, the first step after loading the data is a "Slice" layer, which takes this image, and slices it (i.e. it splits it up) along that axis, thus creating two Top blobs, data and data_p.

>**How to create the data files?**<br />
>There is no single right way to do that. The code from the tutorial is only for the MNIST set, so unless you have the exact same format, you can't use it without changes. You have a couple of possibilities:<br />
>>1.Convert your images to the MNIST-format. Then, the code from the Caffe tutorial works out-of-the-box. It appears that you are trying this - if you need help on that, please be specific: what is "mnisten", include your code, and so on.<br />
>>2.Write your own script to convert the images. This is actually very simple: all you need to do is read the images in your favorite programming language, select the pairs, calculate the labels, and re-save as LMDB. This is definitely the more flexible way.<br />
>>3.Create HDF5 files with multiple Top blobs. This is very simple to do, but will probably be a bit slower than using LMDB.<br />
>What you use is up to you - I'd probably go with HDF5, as this is an easy and very flexible way to start.

>**How to generate the pairs?**<br />
>Now, this is the difficult question here. The code from the tutorial just selects random pairs, which is not really optimal, and will make learning rather slow. You don't just need random pairs, you needmeaningful, difficult, but still solvable pairs. How to do that depends entirely on your dataset.<br />
>A very sophisticated example is presented, in (Radenović, 2016): they use a Siamese network for learning a representation for image retrieval on buildings. They use a Structure-from-Motion (SfM) algorithm to create a 3-D reconstruction of a building, and then sample image pairs from those reconstructions.<br />
>How exactly you create the pairs depends on your data - maybe you are fine with random pairs - maybe you need a sophisticated method.<br />

>**Literature:**<br />
>>F. Radenović, G. Tolias, and O. Chum. "CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples". In: European Conference on Computer Vision (ECCV), 2016. arXiv: 1604.02426.

翻译如下：<br />
您**不需要使用完全相同的格式**，这仅仅是一个教程...<br />
您需要做的是**提供一个或多个数据层（Data Layers)**,这个数据层共有三个top Blobs:data, data_p,sim。<br />
您可以用任何方式来完成这件事，例如LMDB，HDF5或其他格式。<br />
>
**一般解释**<br />
在教程中，他们展示了一种简单的读取图像对（image pairs）的方式：在通道维度（channel dimesion）中连接两个图像。对于灰度空间的图像，输入了两个图像，
对于其中的每一个，我们可以简单的用[1, 1, 28, 28]来表示（1张图，1个通道， 28 x 28的分辨率）。然后我们将他们连接成一个[1, 2, 28, 28]的大小的图像，
并且将他们存储为LMDB。<br />
>
在这个网络中,在读入数据的之后的第一步是一个Slice layer，这一层将图片分开，并且创建两个Top blobs，data和data_p。
>
**如何创建数据文件**<br />
有很多种方法可以办到。教程中的代码仅仅是针对*MNIST*数据集， 所以除非有相同格式的文件，否则是无法在不做任何更改的情况下使用它的。以下有几种可能性：<br />
1.将个人的图像转换成MNIST格式。<br />
2.自己编写脚本转换图像，使用个人习惯的编程语言读取图像，选择对，计算标签，然后重新保存为LMDB，这是更加灵活的方式。<br />
3.使用多个Top blobs创建HDF5文件，这也很简单，但是会比使用LMDB慢一些。<br />
**如何生成对（pairs）**<br />
这是一个比较困难的问题。教程中的代码仅仅是随机选择对，这不是最理想的，并且会使得学习非常慢。你可能不仅仅需要随机的对，更需要一些有意义的、困难的但仍可以解决的对。如何做完全取决于数据集。<br />
（Radenović, 2016）提出了一个非常复杂的例子：他们使用Siamese网络来学习建筑物图像检索的表示。然后使用Structure-from-Motion（SfM）算法对建筑物进行三维重建，然后从这些重建中采样图像对。<br />
如何做取决于自己的数据，也许随机也很好<br />
