由于前面给自己埋了个坑，Miniconda是不带jupyter的啊啊啊啊啊啊啊啊啊啊<br />
于是只能自己填了。<br />
___
Miniconda 64位（python27)的jupyter的安装
------------------------------
### 1 安装jupyter包
首先直接在cmd窗口下，运行conda install jupyter<br />
**强烈建议翻墙**<br />
>
大概要下载一百多兆吧。。等待一下就好了<br />

**PS：安装过程中可能会出现编码的问题**<br />
>
将Python目录下的Lib\site.py前面加上前三行<br />
```
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
```
>

### 2.Jupyter Notebook激活conda环境
**该方法来自csdn的tong_he大神:(https://blog.csdn.net/tong_he/article/details/78813494)**<br />
>
(1)选择要在Jupyter Notebook中激活的环境<br />
```
C:\WINDOWS\system32>conda env list
# conda environments:
#
                         C:\Anaconda3
base                  *  C:\Miniconda-x64
py2                      C:\Miniconda-x64\envs\py2

#################以上步骤是列出conda的环境列表
C:\WINDOWS\system32>activate base

#################以上步骤是选择激活base环境
```
(2)在要激活的环境中安装ipykernel<br />
```
(base) C:\WINDOWS\system32>conda install ipykernel
Solving environment: done

# All requested packages already installed.
```
(3)将选择的conda环境注入jupyter Notebook<br />
**该步骤可能会出现**
```
TraitError: Could not decode 'C:\\Users\\\xc9\xb3\\AppData\\Roaming\\jupyter' for unicode trait 'data_dir' of a KernelSpecManager instance.
```
如上所示的编码问题<br />

**解决方法已经找到，如下:**
```
in file C:\Path\to\Lib\site-packages\jupyter_core\paths.py you need to append .decode(sys.getfilesystemencoding()) at the line end:

in function get_homedir:

homedir = os.path.realpath(homedir).decode(sys.getfilesystemencoding())
in function jupyter_data_dir:

appdata = os.environ.get('APPDATA', None).decode(sys.getfilesystemencoding())
After that jupyter notebook started on my Windows!
```
此方法来自[AndreWin](https://github.com/jupyterhub/jupyterhub/issues/444)<br />
>
为防止网页以后会消失，已经保存成[pdf文件](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/The%20dicision%20of%20the%20problem%20that%20occured%20in%20the%20installation%20of%20jupyter%20on%20Miniconda%20(Windows).pdf)<br />

**接下来**<br />
```
(base) C:\WINDOWS\system32>python -m ipykernel install --user --name base --display-name "Python [conda env:base]"
Installed kernelspec base in C:\Users\沙\AppData\Roaming\jupyter\kernels\base
```
格式： python -m ipykernel install --user --name [要设置的conda中的环境] --display-name [在Jupyter Notebook中要显示的环境变量]<br />
>
(4)conda 打开Jupyter Notebook即可，在打开的UI界面即可看到已加载的内核。
```
(base) C:\WINDOWS\system32>jupyter notebook
```
