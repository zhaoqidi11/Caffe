总算可以在学校的平台上搭建caffe了<br />
具体参考:https://blog.csdn.net/yhaolpz/article/details/71375762 <br />
gpu版本<br />
-------------------
- 系统：ubuntu 16.04
- cuda:9.0
- opencv:2.4.9.1
- cudnn:7.0.5
---------------------
由于学校的平台已经自动安好cuda、opencv、cudnn了，所以不再叙述过程，具体见：
https://blog.csdn.net/yhaolpz/article/details/71375762
### 1.克隆caffe
```
git clone https://github.com/BVLC/caffe.git
```
不成功的话多试几下
### 2.修改Makefile.config文件
#### (1)得到Makefile.config文件
```
cd caffe
sudo cp Makefile.config.example Makefile.config
```
#### (2)修改Makefile.config文件
##### a.应用cudnn
```
将
#USE_CUDNN := 1
修改成： 
USE_CUDNN := 1
```
##### b.使用python接口
```
将
#WITH_PYTHON_LAYER := 1 
修改为 
WITH_PYTHON_LAYER := 1
```
##### c.修改python路径
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
修改为： 
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

### 3.修改caffe目录下的Makefile文件
#### (1)修改NVCCFLAGS
```
将：
NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
替换为：
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)

```
#### (2)修改LIBRARIES
```
将：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
改为：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```
### 4.修改/usr/local/cuda/include/crt下的host_config.h
```
将#error -- unsupported GNU version! gcc versions later than 6 are not supported!注释掉，即
//#error -- unsupported GNU version! gcc versions later than 6 are not supported!
```
### 5.因为个人遇到的一些问题，需要再进行修改
#### (1)解决：Unsupported gpu architecture 'compute_\*'2017
解决方法来自https://blog.csdn.net/jacke121/article/details/55007527<br />
解决方法：修改Makefile.config文件（在caffe目录中）<br />
![caffe-gpu](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/caffe-gpu.png)<br />
将红框框起来的地方注释掉
#### (2)解决：error while loading shared libraries: libcudart.so.9.0: cannot open shared object file: No such file
解决方法来自https://blog.csdn.net/hymanjack/article/details/80199987<br />
解决方法：<br />
```
sudo cp /usr/local/cuda/lib64/libcudart.so.9.0 /usr/local/lib/libcudart.so.9.0 && sudo ldconfig 
sudo cp /usr/local/cuda/lib64/libcublas.so.9.0 /usr/local/lib/libcublas.so.9.0 && sudo ldconfig 
sudo cp /usr/local/cuda/lib64/libcurand.so.9.0 /usr/local/lib/libcurand.so.9.0 && sudo ldconfig
sudo cp /usr/local/cuda/lib64/libcudnn.so.7 /usr/local/lib/libcurand.so.7 && sudo ldconfig
```
### 6.测试
```
sudo make runtest -j8
```
具体截图如下:<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/caffe-gpu2.png)<br />
### 7.编译pycaffe接口
#### (1)编译pycaffe
```
cd caffe
sudo make pycaffe -j8
```
可能会出现错误<br />
```
python/caffe/_caffe.cpp:10:31: fatal error: numpy/arrayobject.h: 没有那个文件或目录
```
解决方法：<br />
```
sudo apt-get install python-numpy
```
#### (2)测试pycaffe
进入python环境，导入caffe:
```
>>> import caffe
```
出现错误①
```
File "<stdin>", line 1, in <module>   ImportError: No module named caffe
```
解决方法:
```
sudo echo export PYTHONPATH="~/caffe/python" >> ~/.bashrc

source ~/.bashrc
```
出现错误②
```
ImportError: No module named skimage.io
```
解决方法:
```
sudo pip install -U scikit-image #若没有安装pip: sudo apt install python-pip
```
在安装过程中可能会出现下载不成功而导致无法安装的情况（错误③），解决方法:
```
sudo apt-get update 
```
之后在安装过程中出现错误④ matplotlib无法安装的情况，解决方法：
```
pip install --upgrade pip #升级pip
```
升级pip之后出现问题⑤：
```
File "/usr/bin/pip", line 9, in <module>
    from pip import main
ImportError: cannot import name main
```
解决方法：
```
sudo gedit /usr/bin/pip
```
修改pip文件（方法来自：https://blog.csdn.net/qq_38522539/article/details/80678412） ，如下：
```
原文：from pip import main 
修改后：from pip._internal import main
```
之后在python环境下import caffe,再次出现错误⑥：
```
ImportError: No module named google.protobuf.internal
```
解决方法（来自https://blog.csdn.net/dgyuanshaofeng/article/details/78151510 )，如下:
```
sudo pip install protobuf
```
之后import caffe成功！
### 8.安装pycharm
参考：https://jingyan.baidu.com/article/60ccbceb4e3b0e64cab19733.html
#### (1)官网下载安装包
地址：https://www.jetbrains.com/pycharm/download/#section=linux 到这里选择安装包
#### (2)选择社区版（Community）下载
#### (3)下载之后解压（extract)
#### (4)进入到bin文件夹下运行pycharm.sh文件
```
sh ./pycharm.sh
```
#### (5)安装成功
#### (6)设定pycharm的python的位置
参考：https://www.cnblogs.com/429512065qhq/p/8663478.html<br />
选择Default　Settings → Project Interpreter → add → 选择python的位置<br />
PS:想要知道python的位置在哪里（https://blog.csdn.net/twt520ly/article/details/79403089）<br />
##### a.查看所有python路径
```
whereis python
```
##### b.查看当前使用的python路径
```
which python
```