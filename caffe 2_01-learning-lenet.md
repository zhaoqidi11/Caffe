首先是准备工作，在cygwin里运行两个sh文件<br />
>
首先是<font color="#660000">根目录下的data\mnist\get_mnist.sh</font><br /> 
>(如果出现'\lr'命令不识别问题的话，用dos2unix转换该文件)<br />
一般网速没问题的话，很快就可以下载好了<br />
接下来要复杂一点<br />
>
修改caffe根目录下的examples\mnist\create_mnist.sh文件<br />
>
将BUILD的目录更改为<br />

```
BUILD=scripts/build/examples/mnist/Release
```
至于为什么BUILD目录会在这个目录下我也不太清楚，只是编译之后就这样了<br />
更改后，在cygwin中切换到**根目录/scripts/build/examples/mnist/Release**下，<br />
我们需要将convert_mnist_data.exe文件转换成.bin文件<br />
（该方法来自https://groups.google.com/forum/#!topic/caffe-users/sXT3TuGt8Nc中的Steven Clark先生）<br />
输入命令<br />
```
ln -s convert_mnist_data convert_mnist_data.bin
```
完成后
切换到caffe的根目录下(一般是有examples，scripts等目录的那个目录）<br />
输入命令<br />
```
bash C:\\caffe\\caffe\\examples\\mnist\\create_mnist.sh
```
即可完成
