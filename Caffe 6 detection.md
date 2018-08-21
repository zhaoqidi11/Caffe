R-CNN是一种先进的探测器，通过精细的Caffe模型对区域建议进行分类。 有关R-CNN系统和模型的完整详细信息，请参阅其项目站点和论文：
>Rich feature hierarchies for accurate object detection and semantic segmentation. Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. CVPR 2014. Arxiv 2013.
在这个例子中，我们通过ImageNet的R-CNN模型的纯Caffe版本进行检测。 R-CNN检测器输出ILSVRC13的200个检测类别的类别分数。 
请记住，这些是原始的和所有SVM分数，因此它们不是概率校准的，也不是跨类别的完全可比的。 请注意，这种现成的模型仅仅是为了方便起见，而不是完整的R-CNN模型。
>
让我们在沙漠中骑自行车骑自行车的人的图像上进行检测（来自ImageNet挑战 - 没有玩笑）。
首先，我们需要区域提案和Caffe R-CNN ImageNet模型：
>
>选择性搜索是R-CNN使用的区域提议者。 selective_search_ijcv_with_python Python模块负责通过选择性搜索MATLAB实现提取提议。 要安装它，请下载模块并将其目录命名为selective_search_ijcv_with_python，在MATLAB中运行演示以编译必要的函数，然后将其添加到PYTHONPATH进行导入。 （如果您准备了自己的区域提案，或者宁愿不打扰此步骤，则detect.py接受图像列表和边界框作为CSV。）

完成后，我们将调用捆绑的detect.py来生成区域提议并运行网络。 有关参数的解释，请执行./dectct.py --help。
>
切换到根目录\examples目录下，在**cygwin**下运行命令<br />
```
mkdir -p _temp
echo `pwd`/images/fish-bike.jpg > _temp/det_input.txt
```
后面由于一些运行问题，修改了examples目录下的_temp目录下的det_input.txt,直接改成绝对路径：
```
C:\caffe\caffe\examples\images\fish-bike.jpg
```
需要下载selective_search_ijcv_with_python，切换到根目录下\examples

```
git clone https://github.com/sergeyk/selective_search_ijcv_with_python
```
然后将selective_search_ijcv_with_python目录复制到caffe的python目录下<br />
然后在python目录下的caffe的detector.py中，用notepad更改，在import caffe后面增加<br />
```
caffe_root = 'C:\\caffe'
import sys
sys.path.insert(0, caffe_root + '\\caffe\\python')
```
然后输入文档中要求的（由于只安装了cpu模式，可以把--gpu这条命令去掉）
```
../python/detect.py --crop_mode=selective_search --pretrained_model=../models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel --model_def=../models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt --gpu --raw_scale=255 _temp/det_input.txt _temp/det_output.h5
```
报错<br />
```
I0821 11:15:57.992342 45092 net.cpp:242] This network produces output fc-rcnn
I0821 11:15:57.992342 45092 net.cpp:255] Network initialization done.
I0821 11:15:58.242365 45092 upgrade_proto.cpp:53] Attempting to upgrade input file specified using deprecated V1LayerParameter: ../models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel
I0821 11:15:58.512852 45092 upgrade_proto.cpp:61] Successfully upgraded file specified using deprecated V1LayerParameter
I0821 11:15:58.532814 45092 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: ../models/bvlc_reference_rcnn_ilsvrc                                             13/bvlc_reference_rcnn_ilsvrc13.caffemodel
I0821 11:15:58.532814 45092 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
W0821 11:15:58.532814 45092 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
CPU mode
Loading input...
selective_search_rcnn({'C:\cygdrive\c\caffe\caffe\examples\images\fish-bike.jpg'}, 'c:\cygwin64\tmp\tmpuo61yp.mat')
Traceback (most recent call last):
  File "../python/detect.py", line 173, in <module>
    main(sys.argv)
  File "../python/detect.py", line 144, in main
    detections = detector.detect_selective_search(inputs)
  File "C:\caffe\caffe\python\caffe\detector.py", line 122, in detect_selective_search
    cmd='selective_search_rcnn'
  File "C:\caffe\caffe\python\selective_search_ijcv_with_python\selective_search.py", line 36, in get_windows
    shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
IOError: [Errno 2] No such file or directory: '/dev/null'

```
然后更改selective_search.py文件，将第36行更改为<br />
```Python
shlex.split(mc), stdout=open('null', 'w'), cwd=script_dirname)
```
**（注意缩进的问题，如果出现问题，在notepad++中，选择视图→显示符号→显示空格与制表符，仔细观察一下然后进行修改）**
在后面还有几行代码需要修改
```Python
retcode = pid.wait()#这行在Windows系统中是没有作用的，所以需要注释掉
```
修改后如下：
```
    #retcode = pid.wait()
    time.sleep(20)#因为Matlab读取图像需要时间（视配置而定），所以需要有这个函数，如果设置的过小，程序会因为没有读取到图像矩阵而退出
    retcode = 0
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")
```
如果time.sleep()中的数值设置过小，会报错，如下
```
CPU mode
Loading input...
selective_search_rcnn({'C:\cygdrive\c\caffe\caffe\examples\images\fish-bike.jpg'}, 'c:\cygwin64\tmp\tmponz1kd.mat')
Traceback (most recent call last):
  File "../python/detect.py", line 173, in <module>
    main(sys.argv)
  File "../python/detect.py", line 144, in main
    detections = detector.detect_selective_search(inputs)
  File "C:\caffe\caffe\python\caffe\detector.py", line 122, in detect_selective_search
    cmd='selective_search_rcnn'
  File "C:\caffe\caffe\python\selective_search_ijcv_with_python\selective_search.py", line 45, in get_windows
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
  File "C:\Miniconda-x64\lib\site-packages\scipy\io\matlab\mio.py", line 135, in loadmat
    MR = mat_reader_factory(file_name, appendmat, **kwargs)
  File "C:\Miniconda-x64\lib\site-packages\scipy\io\matlab\mio.py", line 59, in mat_reader_factory
    mjv, mnv = get_matfile_version(byte_stream)
  File "C:\Miniconda-x64\lib\site-packages\scipy\io\matlab\miobase.py", line 224, in get_matfile_version
    raise MatReadError("Mat file appears to be empty")
scipy.io.matlab.miobase.MatReadError: Mat file appears to be empty
```
这样运行之后，仍然会报错，如下
```
Traceback (most recent call last):
  File "../python/detect.py", line 174, in <module>
    main(sys.argv)
  File "../python/detect.py", line 145, in main
    detections = detector.detect_selective_search(inputs)
  File "C:\caffe\caffe\python\caffe\detector.py", line 125, in detect_selective_earch
    return self.detect_windows(zip(image_fnames, windows_list))
  File "C:\caffe\caffe\python\caffe\detector.py", line 78, in detect_windows
    window_inputs.append(self.crop(image, window))
  File "C:\caffe\caffe\python\caffe\detector.py", line 142, in crop
    crop = im[window[0]:window[2], window[1]:window[3]]
TypeError: slice indices must be integers or None or have an __index__ method
```
查阅之后说是numpy升级之后的问题，解决办法就是把源文件（detector.py）中的出现问题的变量用int(变量名)括起来。
修改如下：
第142行
```
        crop = im[int(window[0]):int(window[2]), int(window[1]):int(window[3])]
```
第176行
```
            context_crop = im[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
```
第179行
```
            crop[int(pad_y):int(pad_y + crop_h), int(pad_x):int(pad_x + crop_w)] = context_crop
```
然后又报错了
```
C:\Miniconda-x64\lib\site-packages\skimage\transform\_warps.py:110: UserWarning:Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifcts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
Traceback (most recent call last):
  File "../python/detect.py", line 174, in <module>
    main(sys.argv)
  File "../python/detect.py", line 167, in main
    df.to_hdf(args.output_file, 'df', mode='w')
  File "C:\Miniconda-x64\lib\site-packages\pandas\core\generic.py", line 1299, i to_hdf
    return pytables.to_hdf(path_or_buf, key, self, **kwargs)
  File "C:\Miniconda-x64\lib\site-packages\pandas\io\pytables.py", line 279, in o_hdf
    complib=complib) as store:
  File "C:\Miniconda-x64\lib\site-packages\pandas\io\pytables.py", line 450, in _init__
    'importing'.format(ex=str(ex)))
ImportError: HDFStore requires PyTables, "No module named tables" problem importng
```
重新安装了一下tables,在cygwin中输入
```
pip install --upgrade tables
```
升级一下tables<br />
最后运行，成功了，提示如下<br />
```
Saved to _temp/det_output.h5 in 0.119 s.
C:\Miniconda-x64\lib\site-packages\skimage\io\_io.py:49: UserWarning: `as_grey` as been deprecated in favor of `as_gray`
  warn('`as_grey` has been deprecated in favor of `as_gray`')
C:\Miniconda-x64\lib\site-packages\skimage\transform\_warps.py:105: UserWarning:The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
  warn("The default mode, 'constant', will be changed to 'reflect' in "
C:\Miniconda-x64\lib\site-packages\skimage\transform\_warps.py:110: UserWarning:Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifcts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
C:\Miniconda-x64\lib\site-packages\pandas\core\generic.py:1299: PerformanceWarnig:
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->mixed,key->block1_values] [items->['predction']]

  return pytables.to_hdf(path_or_buf, key, self, **kwargs)
```

