运行模式：CPU<br />
>
function deprocess net_image(image):<br />
input: caffe格式的图片<br />
output：Python格式的图片（可用于display）<br />
>
**weights:预先加载的已经在Imagenet上训练过的权重**<br />
>
function caffenet(...):<br />
input:data, label(default:None), train(default:True), num_classes(default:1000), classifier_name(default:'fc8'), learn_all(default:False)<br />
output:一个caffe网络，第8个层（全连接层数）的结点是num_classes个，注意是否训练（train是否为真）决定了网络的结构是否包含dropout等层，classifier_name是第8个层的名字，learn_all决定是否要全部训练权重，还是只训练第8层的权重
**dummpy_data：L.DummyData(一个随机数据层）**<br />
**imagenet_net_filename：调用*caffenet*，将dummpy_data作为data输入生成一个网络（train是False）**<br />
**imagenet_net:读取iamgenet_net_filename网络，以及预训练好的权重，测试模式**<br />
>
function style_net(...):<br />
input: train(default:True), learn_all(default:False), subset(default:None):<br />
output: 生成一个新网络，train表示数据来源（训练集还是测试集）和模式（训练还是测试），learn_all表示是否要训练全部。这个新网络的第8层（全连接层的名字已经改成fc8_flickr,个数改成NUM_STYLE_LABELS。<br />
>
**untrained_style_net生成一个新网络（使用style_net())，测试模式，但读入训练集，读取预训练好的权重**<br />
**untrained_style_net.forward()执行前馈过程（运行一个batch）**<br />
**style_data_batch读取这前馈的一批的输入数据**<br />
**style_label_batch读取标签**<br />
>
function disp_preds(...):<br />
input:net,image,labels,k=5, name='ImageNet'<br />
output:net:读入网络，image：读入自己的图像，进行前馈，查看结果（前k(5)个标签）<br />
>
function disp_imagenet_pres(...):<br />
input:net和image<br />
output:disp_preds(net,image,imagenet_labels, name='ImageNet')<br />
>
function disp_style_preds(...):<br />
input:net和image<br />
output:disp_preds(net,image,style_labels, name='style')<br />
