解决方案来自：https://blog.csdn.net/atfuies/article/details/74276668
>
>
原因： stdint.h是c99标准的头文件，vc不支持，所以出现错误“No such file or directory”。<br />

- 解决方法： 
- ①去我的资源中下载msinttypes-r26.zip，地址为：[msinttypes--r26.zip](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/msinttypes--r26.zip)<br />
解压后得到3个文件：inttypes.h，stdint.h，changelog.txt 
- ②找到VC的include目录，一般默认的VC for Python 的include位置为： 
C:\Users\Administrator\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\VC\include <br />
有的用户名不是Administrator，将Administrator替换为你自己的用户名。 
- ③将inttypes.h和stdint.h两个文件放到VC for Python 的include文件中。
