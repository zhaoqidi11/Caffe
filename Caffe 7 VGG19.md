首先解决的是VGG19网络绘制出来的图像（通过draw_net.py）是空白的问题<br />
VGG19的模型及权重下载：https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-vgg_ilsvrc_19_layers_deploy-prototxt<br />
解决办法（来自https://github.com/alexgkendall/caffe-posenet/issues/11）<br />
进入根目录\scripts\build\tools\Release下使用upgrade_net_proto_text.exe将旧的prototxt文件转换成新的prototxt文件<br />
在cmd中输入命令
```
upgrade_net_proto_text.exe old_prototxt.prototxt new_prototxt.prototxt
```
执行结果如下
```

C:\caffe\caffe\scripts\build\tools\Release>upgrade_net_proto_text.exe VGG_ILSVRC_19_layers_deploy.prototxt NEWVGG19.prototxt
I0822 22:06:58.349056 18188 upgrade_proto.cpp:53] Attempting to upgrade input file specified using deprecated V1LayerParameter: VGG_ILSVRC_19_layers_deploy.prototxt
I0822 22:06:58.352056 18188 upgrade_proto.cpp:61] Successfully upgraded file specified using deprecated V1LayerParameter
I0822 22:06:58.352056 18188 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: VGG_ILSVRC_19_layers_deploy.prototxt
I0822 22:06:58.352056 18188 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
W0822 22:06:58.353052 18188 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
I0822 22:06:58.353052 18188 upgrade_net_proto_text.cpp:49] Wrote upgraded NetParameter text proto to NEWVGG19.prototxt

C:\caffe\caffe\scripts\build\tools\Release>
```
转换出来的新文件可被画出网络结构图。
