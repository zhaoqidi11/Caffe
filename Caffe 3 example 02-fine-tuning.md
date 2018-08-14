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
