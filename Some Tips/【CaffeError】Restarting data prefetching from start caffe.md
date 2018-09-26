2018/09/26遇到的一个问题<br />
频繁出现Restarting data prefetching from start. caffe
---
### 问题：Solver文件里test_iter设计的过大
关于solver文件的解释如下<br />
![sovler](https://github.com/meisa233/Caffe/blob/master/Files%20about%20the%20installation%20of%20caffe/Caffe_Solver_File.png)
在我的例子中，测试样本只有250个，而我设置的test_iter=1000，实际上test_iter * batch_size =总的测试样本数量才对，所以这么设定是错的<br />
改为10（在这个例子中我设定的是batch是25）<br />
正确了
