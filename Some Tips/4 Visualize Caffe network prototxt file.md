
# 使用Caffe自带的工具对网络结构进行可视化
## 安装过程
### 1 安装Graphviz
安装方法来自https://blog.csdn.net/u013250416/article/details/72790754<br />
>
在[Graphviz官网](http://www.graphviz.org)中下载msi文件([Windows版本](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)，然后一直安装下去即可<br />
<img src="../Files%20about%20the%20installation%20of%20caffe/Graphviz%20Windows.png" width = 80% div align="center"/>
<br />
安装完毕后，将bin文件夹对应的路径添加到Path系统环境变量中，比如C:\Program Files (x86)\Graphviz2.38\bin。<br />
### 2 安装PyGraphviz
在http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz
下载对应版本<br />
然后使用
```
conda install xxx.whl
```
进行安装<br />
### 3 安装其他组件
#### a. protobuf
```
conda install protobuf
```
#### b. pydotplus
```
conda install pydotplus
```
#### c. graphviz
```
conda install graphviz
```
## 使用说明
见https://blog.csdn.net/dcxhun3/article/details/52237480
在caffe根目录下输入
```
python python\draw_net.py C:\caffe\caffe\examples\siamese\mnist_siamese_train_test.prototxt mynetwrk.png
```
便可以得到一个从左到右的图<br />
![network](../Files%20about%20the%20installation%20of%20caffe/mynetwrk.png)
输入
```
python python\draw_net.py --rankdir TB C:\caffe\caffe\examples\siamese\mnist_siamese_train_test.prototxt mynetwrk_TB.png
```
TB表示从上到下绘制网络结构<br />
![network_TB](../Files%20about%20the%20installation%20of%20caffe/mynetwrk_TB.png)
