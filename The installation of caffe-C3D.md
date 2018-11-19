于是这个C3D的坑是不得不迈出去了<br />
参考：https://blog.csdn.net/kmyfoer/article/details/80060696
### error1:undefined reference to `cv::VideoCapture::VideoCapture()'
解决方案：修改Makefile文件
将<br />
```
LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_video
```
修改为
```
LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_videoio
```
### error2:在测试VideoDataLayerTest/0.TestReadAvi出现错误
```
Unable to stop the stream: Inappropriate ioctl for device
```
查了资料之后，心态崩了，要重新安装Opencv(这里是3.4.0)<br />
解决方法如下：<br />
https://github.com/facebook/C3D/issues/231 <br />
卸载opencv的方法如下（来自https://www.cnblogs.com/txg198955/p/5990295.html）<br />
```
$ sudo make uninstall
$ cd ..
$ sudo rm -r build
$ sudo rm -r /usr/local/include/opencv2 /usr/local/include/opencv /usr/include/opencv /usr/include/opencv2 /usr/local/share/opencv /usr/local/share/OpenCV /usr/share/opencv /usr/share/OpenCV /usr/local/bin/opencv* /usr/local/lib/libopencv*

```
重新安装有ffmpeg的opencv
