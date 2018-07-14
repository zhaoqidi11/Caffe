

Caffe的安装（Windows10+VS2015+Python27)
======================================

准备工作：
------------------------
### 1.下载Miniconda

可以直接在官网下载，地址：https://conda.io/miniconda.html<br />
或者在我的Github上[下载最新版（时至2018/7/12的Windows x64最新版（python27)）](https://github.com/meisa233/Caffe/tree/master/Files%20about%20the%20installation%20of%20caffe)<br />
请注意，在**安装过程**中请选择**将Miniconda加入环境变量**

>除此之外还需要**安装其他组件（pip、six、yaml、numpy)**:<br />
注意这里我们是要对**Miniconda安装目录中的Python**安装其他组件,即C:\Miniconda-x64中（由于后面的安排，特意将Miniconda安装在这个文件夹中），后面解释。<br />

>>**(1)首先安装pip：**<br />
在cmd窗口中（为了保险起见，请**选择管理员权限运行**，切换到C:\Miniconda-x64\Scripts下，输入**easy_install.exe pip**,一般没有什么问题的话，就会安装成功。<br />

>>**(2)安装six**<br />
>>>一般来说，运行Miniconda下的python之后，直接输入**pip install six**即可成功，但有时候会失败（原因未知，网络问题？），这时候需要到[Pypi官方网站](https://pypi.org/)下载相应的包，注意是whl为后缀的，我们以six组件为例<br />
>>>打开Pypi的官方网站，在搜索栏输入six<br />
>>>![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/1.png "1.png")<br />
点开six，可以看到<br />
>>>![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/2.png)<br />
>>>我们选择Download files，可以看到这个界面<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/3.png)<br />
>>>我们选择six-1.11.0-py2.py3-none-any.whl，然后会下载下来这个文件，**将这个文件放到C:\Miniconda-x64\Scripts下**，为了使pip命令可以直接运行，我们**将pip的路径加入环境变量（path）**<br />
>>>在本例中是“C:\Miniconda-x64\Scripts”，将这个变量加入**path（系统变量）**，然后运行**cmd**，在cmd中输入“cd /d C:\Miniconda-x64\Scripts”,将当前目录更改到这个目录下<br />
>>>然后输入“pip install six-1.11.0-py2.py3-none-any.whl”，即可安装成功。<br />
>>>注意在这个例子中，只有一个版本，通常一个组件（module）是有很多个版本的，如图：<br />
>>>![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/4.png)<br />
因此我们要选择符合**系统**以及符合**Python版本**的module，在这里我们是windows、64位、python27。<br />

>>**(3)安装yaml: pip install pyyaml**

>>**(4)安装numpy:pip install numpy**


### 2.安装VS2015

这里用的 官网离线下载镜像（社区版）：http://download.microsoft.com/download/B/4/8/B4870509-05CB-447C-878F-2F80E4CB464C/vs2015.com_chs.iso

如有需要可以看[其他版本](https://github.com/meisa233/Caffe/blob/master/VS2015%20Download.md)<br />

* 注意：默认安装VS2015的时候是**不会包含C++编译器**的，所以一定要选择**自定义安装**，选择**C++编译器（即C++开发工具）**，以防万一选上**Python和SDK等其他组件**

### 3.安装CMake

这里安装的最新版3.12.0-rc3-windows-x64(时至2018/7/12)，官网下载地址：https://cmake.org/download/<br />
或者在我的Github上[下载](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/cmake-3.12.0-rc3-win64-x64%20(1).7z)<br />
**网上的资料要求CMake的版本必须在3.4以上（没有验证这个问题，直接安装的最新版）**<br />
下载之后，解压到C盘目录下，本例是：“C:\cmake-3.12.0-rc3-win64-x64”

**将CMake解压后的路径\bin，即“C:\cmake-3.12.0-rc3-win64-x64\cmake-3.12.0-rc3-win64-x64\bin”写入环境变量path（系统变量）”

### 4.下载依赖库

时至2018/7/12，目前最新版本的依赖库是libraries_v140_x64_py27_1.1.0.tar.bz2，可以从https://github.com/willyd/caffe-builder/releases
下载到。<br />
下载后将这个文件放到**“C:\Users\沙\.caffe\dependencies\download”,即“C:\Users\用户名\.caffe\dependencies\download”下<br />
（由于安装完之后才写的这个文档，忘记具体是什么情况下了，在dependencies下也有这个tar.bz2的文件以及该文件的压缩包）**<br/>

### 5. 安装Git Credential Manager for Windows

可以从https://github.com/Microsoft/Git-Credential-Manager-for-Windows/tags这里下载，本例用的1.17预览版，你可以选择更新的版本。

本例用的版本的[下载地址](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/GCMW-1.17.0-preview.2.exe)<br />

### 6.建议安装notepad++



开始安装：
------------------------
### 1.下载caffe源码
使用 **GCMW（Git Credential Manager for Windows）进行克隆**，打开**Git CMD（直接在开始菜单找或者搜索）**<br />

进入**需要克隆的本地目录**，输入**git clone https://github.com/BVLC/caffe.git**。
如下：
```
C:\caffe>git clone https://github.com/BVLC/caffe.git
C:\caffe>cd caffe
C:\caffe\caffe>git checkout windows

```
下载过程会比较漫长，完成以上操作即可。

### 2.编辑build_win.cmd

建议使用**notepad++**编辑build_win.cmd（在caffe\scripts\下）,具体编辑过程见[这里](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/build_win%202.cmd)

### 3.执行build_win.cmd
建议使用**管理员权限**打开cmd，然后到该目录下执行**build_win.cmd**<br />
**注意：如果中间有任何执行失败的地方，请删除caffe\build目录下的全部内容，以及caffe\scripts\build目录下的全部内容**

### 4.编译pycaffe
用VS2015打开caffe.sln，右击
首先需要修改项目的一些**属性**：<br />
在这里我们使用**CPU模式**，需要进行一些修改：<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/6.png)<br />

>右击pycaffe→属性<br />
>>（1）左侧的**配置属性**里选择“**VC++目录**”：**包含目录**填入“Python目录\libs”和“Python目录\include”,本例中是“C:\Miniconda-x64\libs”和“C:\Miniconda-x64\include”<br />
>>（2）**库目录**填入”Python目录\Lib”，本例中是“C:\Miniconda-x64\Lib”、“C:\Miniconda-x64\libs”和“C:\Miniconda-x64\include”<br />
>>(3)**C/C++的附加包含目录**内容如下（具体记不清楚了只好截图）：<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/5.png)<br />
>>主要是**要填入：“Python目录\Lib”、“Python目录\include”以及“Python目录\Lib\site-packages\numpy\core\include”**
>>(4)**链接器的常规的附加库目录**填入以下内容：C:\caffe\caffe\scripts\build\lib\Release

接下来先编译运行caffe.cpp

在编译之前，我们需要对解决方案的属性进行一下配置，如下图<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/7.png)
**将pycaffe设定为单启动项目**

**然后VS2013里面pycaffe项目点击生成。生成之后如果显示这个无法打开_caffe.pyd文件，就说明生成成功了**

### 5.安装protobuf
>(1)下载源码<br />
>>到[Google](https://github.com/google/protobuf/releases)上下载Python版源码和Win32.zip<br />
>>或者到[这里下载Python版protobuf源码](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/protobuf-python-3.6.0.zip)<br />
>>以及到[这里下在win32.zip](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/protoc-3.6.0-win32%20(1).zip)<br />
>>**解压protoc-3.6.0-win32 (1).zip→找到bin目录下的protoc.exe→将这个protoc.exe放到python源码包解压后的src目录下→然后执行python setup.py→python setup.py test→python setup.py install**
>>**如果在安装过程中报错：UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 7: ordi，可以进行如下操作:<br />
>>打开Python目录/Lib文件夹下的site.py文件,在文件开头加入三行：**<br />
```
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
```

该方法来源：https://blog.csdn.net/Ehcoing/article/details/78983339
### 6.安装其他的Python包(请按照顺序来）
(1)Cython>=0.19.2<br />
[官网下载地址](https://pypi.org/project/Cython/),[时至2018/7/14的python27win64版](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/Cython-0.28.4-cp27-cp27m-win_amd64.whl)
(2)numpy-mkl版(版本>=1.7.1)<br />
[官网下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)<br />
或者<br />
[numpy-1.14.5+mkl-cp27-cp27m-win_amd64.7z.001](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/numpy-1.14.5%2Bmkl-cp27-cp27m-win_amd64.7z.001)<br />
[numpy-1.14.5+mkl-cp27-cp27m-win_amd64.7z.002](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/numpy-1.14.5%2Bmkl-cp27-cp27m-win_amd64.7z.002)<br />
具体安装教程见：https://www.jianshu.com/p/6b2a50386701
(3)scipy>=0.13.2
[Pypi下载地址](https://pypi.org/project/scipy/#files)<br/>
[镜像](https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)<br />
[scipy-1.1.0rc1-cp27-none-win_amd64.whl分卷1](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scipy-1.1.0rc1-cp27-none-win_amd64.7z.001)<br />
[scipy-1.1.0rc1-cp27-none-win_amd64.whl分卷2]((https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scipy-1.1.0rc1-cp27-none-win_amd64.7z.002)<br />
(4)scikit-image>=0.9.3
安装scikit-image需要先安装其他包:
a.**six>=1.10.0**<br />
在准备工作中的1的(1)已经安装好了，不做赘述<br />
___
b.**dask[array]>=0.9.0**<br />
[dask-0.18.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/dask-0.18.0-py2.py3-none-any.whl)<br />
___
c.**Cython>=0.23**<br />
在这一步的第（1）步已经安装好了，不做赘述
___
d.**SciPy>=0.9**<br />
第（3）步已经安装好了
___
e.**numpydoc>=0.6**<br />
>**e.a sphinx>=1.2.3**<br />
>要安装sphinx，需要安装其他包<br />
>>**e.a.a Jinja2>=2.3**<br />
>>要安装Jinja2，需要先安装**MarkupSafe>=0.23**<br />
>>[官方下载地址](https://pypi.org/project/MarkupSafe/#files)<br />
>>[MarkupSafe-1.0](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Jinja2/MarkupSafe-1.0-cp27-cp27m-win_amd64.whl)<br />
---
>>然后安装**Jinja2**,[官方下载地址](https://pypi.org/project/Jinja2/#files)<br />
>>[Jinja2-2.10-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Jinja2/Jinja2-2.10-py2.py3-none-any.whl)<br />
---
>>**e.a.b colorama>=0.3.5**<br />
>>[官方下载地址](https://pypi.org/project/colorama/#files)<br />
>>[colorama-0.3.9-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/colorama-0.3.9-py2.py3-none-any.whl)<br />
---
>>**e.a.c Babel!=2.0且>=1.3**<br />
>>要安装Babel需要先安装**pytz>=0a**
>>[官方下载地址](https://pypi.org/project/pytz/#files)<br />
>>[pytz-2018.5-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Babel/pytz-2018.5-py2.py3-none-any.whl)<br />
---
>>然后安装Babel<br />
>>[官方下载地址](https://pypi.org/project/Babel/#files)<br />
>>[Babel-2.6.0-py2-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Babel/Babel-2.6.0-py2-none-any.whl)<br />
---
>>**e.a.d typing (python_version<3.5)**<br />
>>[官方下载地址](https://pypi.org/project/typing/#files)<br />
>>[typing-3.6.4-py2-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/typing-3.6.4-py2-none-any.whl)<br />
---
>>**e.a.e docutils>=0.11**<br />
>>[官方下载地址](https://pypi.org/project/docutils/#files)<br />
>>[docutils-0.14-py2-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/docutils-0.14-py2-none-any.whl)<br />
---
>>**e.a.f alabaster<0.8,>=0.7**<br />
>>[官方下载地址](https://pypi.org/project/alabaster/#files)<br />
>>[alabaster-0.7.11-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/alabaster-0.7.11-py2.py3-none-any.whl)<br />
---
>>**e.a.g snowballstemmer>=1.1**<br />
>>[官方下载地址](https://pypi.org/project/snowballstemmer/#files)<br />
>>[snowballstemmer-1.2.1-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/snowballstemmer-1.2.1-py2.py3-none-any.whl)<br />
---
>>**e.a.h Pygments>=2.0**<br />
>>[官方下载地址](https://pypi.org/project/Pygments/#files)<br />
>>[Pygments-2.2.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Pygments-2.2.0-py2.py3-none-any.whl)<br />
---
>>**e.a.i packaging**<br />
>>要安装packaging，需要先安装**pyparsing>=2.0.2**
>>[官方下载地址](https://pypi.org/project/pyparsing/#files)<br />
>>[pyparsing-2.2.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/packaging/pyparsing-2.2.0-py2.py3-none-any.whl)<br />
---
>>然后安装packaging<br />
>>[官方下载地址(https://pypi.org/project/packaging/#files)<br />
>>[packaging-17.1-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/packaging/packaging-17.1-py2.py3-none-any.whl)<br />
---
>>**e.a.j sphinxcontrib_websupport**<br />
>>[官方下载地址](https://pypi.org/project/sphinxcontrib-websupport/#files)<br />
>>[sphinxcontrib_websupport-1.1.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/sphinxcontrib_websupport-1.1.0-py2.py3-none-any.whl)<br />
---
>>**e.a.k imagesize**<br />
>>[官方下载地址](https://pypi.org/project/imagesize/#files)<br />
>>[imagesize-1.0.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/imagesize-1.0.0-py2.py3-none-any.whl)<br />
---
>**最后安装sphinx**<br />
>[官方下载地址](https://pypi.org/project/Sphinx/#files)<br />
>[Sphinx-1.7.5-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/Sphinx/Sphinx-1.7.5-py2.py3-none-any.whl)<br />
---
>**安装numpydoc**<br />
>[官方下载地址](https://pypi.org/project/numpydoc/#files)<br />
>[numpydoc-0.8.0.tar.gz](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/numpydoc/numpydoc-0.8.0.tar.gz)<br />
<br />
f.**PyWavelets>=0.4**<br />
[官方下载地址](https://pypi.org/project/PyWavelets/#files)<br />
[PyWavelets-0.5.2-cp27-cp27m-win_amd64.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/PyWavelets-0.5.2-cp27-cp27m-win_amd64.whl)<br />
---
g.**cloudpickle>=0.2.1**<br />
[官方下载地址](https://pypi.org/project/cloudpickle/#files)<br />
[cloudpickle-0.5.3-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/cloudpickle-0.5.3-py2.py3-none-any.whl)<br />
---
h.**NetworkX>=1.8**<br />
[官方下载地址](https://pypi.org/project/networkx/#files)<br />
[networkx-2.1-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/networkx-2.1-py2.py3-none-any.whl)<br />
---
i.**Pillow>=4.3.0**<br />
[官方下载地址](https://pypi.org/project/Pillow/#files)<br />
[Pillow-5.2.0-cp27-cp27m-win_amd64.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/Pillow-5.2.0-cp27-cp27m-win_amd64.whl)<br />
---
j.**toolz>=0.7.3**<br />
[官方下载地址](https://pypi.org/project/toolz/#files)<br />
[toolz-0.9.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/scikit_image/toolz-0.9.0-py2.py3-none-any.whl)<br />
---
**最后安装scikit-image**<br />
[官方下载地址]()<br />
[]()<br />
---
(5)matplotlib>=1.3.1<br />
a.**python-dateutil>=2.1**<br />
[官方下载地址](https://pypi.org/project/python-dateutil/#files)<br />
[python_dateutil-2.7.3-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/matplotlib/python_dateutil-2.7.3-py2.py3-none-any.whl)<br />
---
b.**backports.functools-lru-cache**<br />
[官方下载地址](https://pypi.org/project/backports.functools_lru_cache/#files)<br />
[backports.functools_lru_cache-1.5-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/matplotlib/backports.functools_lru_cache-1.5-py2.py3-none-any.whl)<br />
---
c.**cycler>=0.10**<br />
[官方下载地址](https://pypi.org/project/Cycler/#files)<br />
[cycler-0.10.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/matplotlib/cycler-0.10.0-py2.py3-none-any.whl)<br />
---
d.**kiwisolver>=1.0.1**<br />
[官方下载地址](https://pypi.org/project/kiwisolver/#files)<br />
[kiwisolver-1.0.1-cp27-cp27m-win_amd64.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/matplotlib/kiwisolver-1.0.1-cp27-cp27m-win_amd64.whl)<br />
---
**最后安装matplotlib**<br />
(6)ipython>=3.0.0<br />
a.**prompt-toolkit>=1.0.4**<br />
需要安装**wcwidth**<br />
[官方下载地址](https://pypi.org/project/wcwidth/#files)<br />
[wcwidth-0.1.7-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/prompt_toolkit/wcwidth-0.1.7-py2.py3-none-any.whl)<br />
---
最后安装**prompt-toolkit**<br />
[官方下载地址](https://pypi.org/project/prompt/#files)<br />
[prompt_toolkit-1.0.15-py2-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/prompt_toolkit/prompt_toolkit-1.0.15-py2-none-any.whl)<br />
---
b.**pickleshare**<br />
b.a**pathlib2**<br />
b.a.a **sandir**
[官方下载地址](https://pypi.org/project/scandir/#files)<br />
[scandir-1.7-cp27-cp27m-win_amd64.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/pickleshare/pathlib/scandir-1.7-cp27-cp27m-win_amd64.whl)<br />
---
然后安装**pathlib2**<br />
[官方下载地址](https://pypi.org/project/pathlib2/#files)<br />
[pathlib2-2.3.2-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/pickleshare/pathlib/pathlib2-2.3.2-py2.py3-none-any.whl)<br />
---
然后安装**pickleshare**<br />
[官方下载地址](https://pypi.org/project/pickleshare/#files)<br />
[pickleshare-0.7.4-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/pickleshare/pickleshare-0.7.4-py2.py3-none-any.whl)<br />
---
c.**win-unicode-console>=0.5**<br />
[官方下载地址](https://pypi.org/project/win_unicode_console/#files)<br />
[win_unicode_console-0.5-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/win_unicode_console-0.5-py2.py3-none-any.whl)<br />
---
d.**backports.shutil-get-terminal-size**<br />
[官方下载地址](https://pypi.org/project/backports.shutil_get_terminal_size/#files)<br />
[backports.shutil_get_terminal_size-1.0.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/backports.shutil_get_terminal_size-1.0.0-py2.py3-none-any.whl)<br />
---
e.**simplegeneric>0.8**<br />
[官方下载地址](https://pypi.org/project/simplegeneric/#files)<br />
[simplegeneric-0.8.1-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/simplegeneric-0.8.1-py2.py3-none-any.whl)<br />
---
f.**traitlets>=4.2**<br />
f.a**ipython-genutils**<br />
[官方下载地址](https://pypi.org/project/ipython_genutils/#files)<br />
[ipython_genutils-0.2.0-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/traitlets/ipython_genutils-0.2.0-py2.py3-none-any.whl)<br />
最后安装**traitlets**<br />
[官方下载地址](https://pypi.org/project/traitlets/#files)<br />
[traitlets-4.3.2-py2.py3-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/traitlets/traitlets-4.3.2-py2.py3-none-any.whl)<br />
---
最后安装**ipython**<br />
[官方下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/#jupyter)<br />
[ipython-5.7.0-py2-none-any.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/ipython/ipython-5.7.0-py2-none-any.whl)<br />
---
(7)h5py>=2.2.0<br />
[官方下载地址](https://pypi.org/project/h5py/#files)<br />
[h5py-2.8.0-cp27-cp27m-win_amd64.whl](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/h5py-2.8.0-cp27-cp27m-win_amd64.whl)<br />
