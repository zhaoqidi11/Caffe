好吧又是Windows的坑<br />
本来以为下载的GCMW可以直接运行.sh文件的，结果我真的是太天真<br />
>

于是我们用**Cygwin**这个软件来解决问题<br />
>
Crgwin的安装和配置
-------------------------
### 1.官网
http://www.cygwin.com/
### 2.下载
我们可以选择图中黑框中的这个链接，进行下载<br />
![image](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/Cygwin/11.png)<br />
### 3.安装
#### 3.1
安装的时候选择**install from Internet**<br />
#### 3.2
选择一个安装目录(Local Package Directory)<br />
#### 3.3
在Seclect Connection Type这一步，我们选择"Direct Connection"<br />
#### 3.4
选择一个镜像，当时我是随便选的，选的第一个（速度还是挺快的）<br />
>
网友推荐中国镜像：<br />
(http://cygwin.cn/pub/)或者(http://mirrors.ustc.edu.cn)<br />
>
#### 3.5
在Select Packages这一步，我选的是mingw64-x86_64-gcc-g++和bash（需要在Search框内输入进行搜索）<br />
>
除此之外，建议找到**wget**，因为后面执行一些下载包的命令可能会用到<br />
### 4.执行terminal
安装完毕后,从开始菜单找到Cygwin64 Terminal，运行即可<br />
