

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
>右击pycaffe→属性<br />
>>左侧的**配置属性**里选择“**VC++目录**”：填入“Python目录\libs”和“Python目录\include”。

