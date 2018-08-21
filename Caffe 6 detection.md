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
