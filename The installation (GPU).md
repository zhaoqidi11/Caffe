总算可以在学校的平台上搭建caffe了<br />
具体参考:https://blog.csdn.net/yhaolpz/article/details/71375762 <br />
gpu版本<br />
-------------------
- 系统：ubuntu 16.04
- cuda:9.0
- opencv:2.4.9.1/3.4.0
- cudnn:7.0.5
---------------------
## 装opencv
重新申请了一个，发现opencv没了（exm？？）
于是需要装一下opencv，安装方法来自https://blog.csdn.net/qq473179304/article/details/79444609<br />
到官网：http://opencv.org/releases.html ，下载3.4.0的opencv，解压到随意一个位置<br />
在opencv下新建一个文件夹build<br />
```
cd build  
  
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..  
#如果cmake命令不存在，用sudo apt-get install cmake来安装cmake  
make -j8  #编译  
```
编译完成后，安装
```
sudo make install #安装  
```
安装完成后通过查看opencv版本验证是否安装成功：
```
pkg-config --modversion opencv 
```
**python2.7 装 opencv** <br />
```
sudo pip install opencv-python
```
(也有人用python-opencv试过……会出现问题）<br />
由于学校的平台已经自动安好cuda、opencv、cudnn了，所以不再叙述过程，具体见：
https://blog.csdn.net/yhaolpz/article/details/71375762
### 1.克隆caffe
```
git clone https://github.com/BVLC/caffe.git
```
不成功的话多试几下<br />
PS:如果显示没有git这个命令，输入以下内容
```
sudo apt-get install git
```
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
##### d.如果使用的opencv版本是3修改如下
```
将：  
#OPENCV_VERSION := 3   
修改为：   
OPENCV_VERSION := 3  
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
#### (3)为防止出现问题，安装一下依赖
解决方法来自:https://blog.csdn.net/bone_ace/article/details/80645522<br />
```
sudo apt-get install git
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev
libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install python-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
#### (2)解决：error while loading shared libraries: libcudart.so.9.0: cannot open shared object file: No such file
解决方法来自https://blog.csdn.net/hymanjack/article/details/80199987<br />
解决方法：<br />
```
sudo cp /usr/local/cuda/lib64/libcudart.so.9.0 /usr/local/lib/libcudart.so.9.0 && sudo ldconfig 
sudo cp /usr/local/cuda/lib64/libcublas.so.9.0 /usr/local/lib/libcublas.so.9.0 && sudo ldconfig 
sudo cp /usr/local/cuda/lib64/libcurand.so.9.0 /usr/local/lib/libcurand.so.9.0 && sudo ldconfig
sudo cp /usr/local/cuda/lib64/libcudnn.so.7 /usr/local/lib/libcurand.so.7 && sudo ldconfig
```
#### (3)提示 make: protoc: Command not found，好吧，需要安装protoc-c
```
sudo apt-get install protobuf-c-compiler protobuf-compiler
````
#### (4)./include/caffe/common.hpp:4:32: fatal error: boost/shared_ptr.hpp: 没有那个文件或目录
解决方法来自：https://blog.csdn.net/lwgkzl/article/details/77657933<br />
```
sudo apt-get install --no-install-recommends libboost-all-dev
```
#### (5)./include/caffe/common.hpp:5:27: fatal error: gflags/gflags.h: No such file or directory
解决方法来自：https://blog.csdn.net/u012576214/article/details/68947893
```
sudo apt-get install libgflags-dev
```
#### (6)fatal error: gflags/gflags.h:没有那个文件或目录
解决方法来自：https://www.cnblogs.com/zjutzz/p/5716453.html?utm_source=itdadao&utm_medium=referral
```
sudo apt-get install libgoogle-glog-dev
```
#### (7)fatal error: caffe/proto/caffe.pb.h: No such file or directory
解决方法来自:https://blog.csdn.net/lanchunhui/article/details/58245582
```
$ protoc src/caffe/proto/caffe.proto --cpp_out=.
$ sudo mkdir include/caffe/proto
$ sudo mv src/caffe/proto/caffe.pb.h include/caffe/proto
```
#### (8)fatal error: google/protobuf/stubs/common.h: no such file or directory
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
#### (9)/usr/bin/ld: cannot find -lsnappy
```
sudo apt-get install libsnappy-dev
```
### 6.安装与测试
安装，在caffe目录下执行
```
make all -j8
```
如果失败了，先make clean然后再make all -j8<br />
测试
```
sudo make runtest -j8
```
具体截图如下:<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/caffe-gpu2.png)<br />
### 7.编译pycaffe接口
#### (1)编译pycaffe
```
#caffe根目录下
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
sudo echo export PYTHONPATH="~/caffe/python" >> ~/.bashrc#注意引号里面的caffe是caffe的根目录地址

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
**有可能会出现“Could not find a version that satisfies the requirement scikit-image (from versions:) No matching distribution found for scikit-image”，再试几次就好**<br />
**如果下载过程中出现“ReadTimeoutError：HTTPSConnectionPool(host='files.pythonhosted.org', port=443):Read timed out.”，使用清华镜像可以解决问题，使用方法：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package**（来源见https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)<br />
在安装过程中可能会出现下载不成功而导致无法安装的情况（错误③），解决方法:
```
sudo apt-get update 
```
之后在安装过程中出现错误④ matplotlib无法安装的情况，解决方法：<br />
升级pip<br />
**2018/10/08更新：pip更新方法见https://www.jb51.net/article/138426.htm<br />
简述：到pypi下载pip的tar.gz包，解压后，在该目录下python setup.py install<br />
但是注意，这个时候安装matplotlib仍然会出现问题，所以先安装matplotlib指定版本 ：sudo pip install matplotlib==2.2.3<br />
然后再安装scikit-image**<br />
>```
>pip install --upgrade pip #升级pip
>```

>升级pip之后出现问题⑤：
>```
>File "/usr/bin/pip", line 9, in <module>
>    from pip import main
>ImportError: cannot import name main
>```
>解决方法：
>```
>sudo gedit /usr/bin/pip
>```
>修改pip文件（方法来自：https://blog.csdn.net/qq_38522539/article/details/80678412） ，如下：
>```
>原文：from pip import main 
>修改后：from pip._internal import main
>```
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
##### c.如何运行pycharm
参考：https://blog.csdn.net/yimixgg/article/details/79442175
进入到pycharm的目录下的bin文件夹
```
bash ./pycharm.sh
```
### 9.安装VSCode
参考https://www.cnblogs.com/lzpong/p/6145511.html <br />
因为pycharm有点翻车，于是改用VSCode
#### (1)官网下载压缩包
访问Visual Studio Code官网：<br />
https://code.visualstudio.com/docs?dv=linux64 <br />
#### (2)解压
可以直接右键解压
#### (3)移动到/usr/local/目录
```
mv SCode-linux-x64 /usr/local
```
#### (4)给可执行权限
```
chmod +x /usr/local/VSCode-linux-x64/code
```
#### (5)复制VScode图标文件到/usr/share/icons/目录
```
cp /usr/local/VSCode-linux-x64/resources/app/resources/linux/code.png /usr/share/icons/
```
#### (6)创建启动器，在/usr/share/applications/目录
```
sudo vim /usr/share/applications/VSCode.desktop
```
然后输入以下文本<br />
```
[Desktop Entry]
Name=Visual Studio Code
Comment=Multi-platform code editor for Linux
Exec=/usr/local/VSCode-linux-x64/code
Icon=/usr/share/icons/code.png
Type=Application
StartupNotify=true
Categories=TextEditor;Development;Utility;
MimeType=text/plain;
```
保存退出（先按esc键，然后输入:wq)<br />
然后可以将它复制到桌面<br />
#### (7)安装python
打开VS Code，在左侧有个图标![icon_VSCODE](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/QQ%E5%9B%BE%E7%89%8720180928013937.png)
#### (8)安装pylint可能需要输入的命令
```
sudo python -m pip install -U "pylint<2.0.0" --ignore-installed enum34 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
#### (9)出现问题“Visual Studio Code is unable to watch for file changes in this large workspace" (error ENOSPC)
参考https://code.visualstudio.com/docs/setup/linux#_visual-studio-code-is-unable-to-watch-for-file-changes-in-this-large-workspace-error-enospc
