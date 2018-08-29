在Caffe的文档里这么写：
group (g) [default 1]: If g > 1, we restrict the connectivity of each filter to a subset of the input. <br />
Specifically, the input and output channels are separated into g groups, <br />
and the ith output group channels will be only connected to the ith input group channels.

group (g) [default 1]:如果g > 1，我们将每个过滤器的连通性限制到输入的一个子集。具体来说，输入和输出通道被分成g组，第i个输出组通道只连接到第i个输入组通道。
