Slience Layer的作用
---
#### 源码
```
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}
```
解释，来自https://stackoverflow.com/questions/42172871/explain-silence-layer-in-caffe<br />
>The use of this layer is simply to avoid that the output of unused blobs is reported in the log. Being an output manager layer, it is obviously zero its gradient.

>For instance, let us assume we are using AlexNet and we change the bottom of the 'fc7' layer to 'pool5' instead of 'fc6'. If we do not delete the 'fc6' blob declaration, this layer is not used anymore but its ouput will be printed in stderr: it is considered as an output of the whole architecture. If we want to keep 'fc6' for some reasons, but without showing its values, we can use the 'SilenceLayer'.
中文解释如下：<br />
该层将不会被使用并且在日志中输出，且梯度为0。<br />
>
官网介绍：http://caffe.berkeleyvision.org/tutorial/layers/silence.html
