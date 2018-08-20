### Introduction
While Caffe is made for deep networks it can likewise represent "shallow" models like logistic regression for classification. <br />
We'll do simple logistic regression on synthetic data that we'll generate and save to HDF5 to feed vectors to Caffe. Once that model<br />
is done, we'll add layers to improve accuracy. That's what Caffe is about: define a model, experiment, and then deploy.<br />
虽然Caffe是用于深度网络，但它同样可以表示“浅”模型，如分类的逻辑回归。 我们将对我们将生成的合成数据进行简单的逻辑回归，<br />
并保存到HDF5格式的数据集中，用以输入到Caffe中。 一旦该模型完成，我们将添加层以提高准确性。 <br />
这就是Caffe的意义：定义一个模型，实验，然后部署。<br />
>
Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative features and 2 noise features.<br />

创建一个10000*4的数据，用于二元分类，具有2个信息特征和2个噪声特征。
