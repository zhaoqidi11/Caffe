Fine-tuning a Pretrained Network for Style Recognition<br />
微调一个预先训练的风格识别网络<br />
预训练Caffe Network并且微调参数<br />

这种方法的优点是，由于预先训练的网络是在大量图像上学习的，中间层捕获了一般视觉外观的“语义”。<br />
可以把它看作一个非常强大的通用视觉特性，您可以将其视为一个黑盒。除此之外，要想在目标任务中<br />
获得良好的性能，只需要相对较少的数据。<br />

需要准备数据，（1）使用提供的Shell脚本获取ImageNet ilsvrc预训练模型；（2）下载完整的Filckr样式数据集的子集<br />
（3）将下载的Filckr数据集编译到Caffe可以使用的数据库中
