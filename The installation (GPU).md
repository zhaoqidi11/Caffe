总算可以在学校的平台上搭建caffe了<br />
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
#### 6.测试
```
sudo make runtest -j8
```
具体截图如下:<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/caffe-gpu2.png)<br />
