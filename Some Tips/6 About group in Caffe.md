在Caffe的文档里这么写：
group (g) [default 1]: If g > 1, we restrict the connectivity of each filter to a subset of the input. <br />
Specifically, the input and output channels are separated into g groups, <br />
and the ith output group channels will be only connected to the ith input group channels.

group (g) [default 1]:如果g > 1，我们将每个过滤器的连通性限制到输入的一个子集。具体来说，输入和输出通道被分成g组，第i个输出组通道只连接到第i个输入组通道。<br />
caffe Convolution层的convolution_param参数字典中有一个group参数，其意思是将对应的输入通道与输出通道数进行分组，比如输入数据大小为
90x100x100x32 90是数据批大小 100x100是图像数据shape，32是通道数，要经过一个3x3x48的卷积，group默认是1，就是全连接的卷积层，
如果group是2，那么对应要将输入的32个通道分成2个16的通道，将输出的48个通道分成2个24的通道。对输出的2个24的通道，第一个24通道与输入的第一个16通道进行全卷积，第二个24通道与输入的第二个16通道进行全卷积。极端情况下，输入输出通道数相同，比如为24，group大小也为24，那么每个输出卷积核，只与输入的对应的通道进行卷积。
