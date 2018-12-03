## 使用版本**C3D-v1.1**
### 1.处理数据集(视频→帧)
教程来源：http://vra.github.io/2016/03/03/c3d-use/
需要使用脚本对数据集进行分类<br />
脚本来源于：https://github.com/meisa233/CNN/blob/master/C3DTest.py<br />
```Python
from glob import glob
import cv2
import os
# Extract the frame from videos from UCF-101 dataset and convert them to images
# The name of converted images are "%06d.jpg" and notice that the name of image into C3D (in caffe) from 1 !!
if __name__ == '__main__':
    AllFolders = glob('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/UCF-101/*') #Get All Folders in this path
    # print AllFolders
    # os.chdir('/data/C3D/C3D-v1.1')
    for i in AllFolders:#For every folder in this path
        VideoPath = [i, "*.avi"]
        AllVideos = glob("/".join(VideoPath))
        os.mkdir('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/%s' % i.split('/')[-1])#Create folders
        for j in AllVideos:
            os.chdir('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/%s' % i.split('/')[-1])
            i_Video = cv2.VideoCapture(j)
            success = True
            framenumber = 1
            os.mkdir(j.split('/')[-1].split('.')[0])
            os.chdir(j.split('/')[-1].split('.')[0])
            while(success):
                success, frame = i_Video.read()
                cv2.imwrite('%06d.jpg' % framenumber, frame)
                framenumber = framenumber + 1
```
### 2.提取特征
使用build/tools文件夹下的extract_image_features.bin<br />
用法如下<br />
```
 extract_image_features.bin <feature_extractor_prototxt_file> 
<c3d_pre_trained_model> <gpu_id> <mini_batch_size> 
<number_of_mini_batches> <output_prefix_file> <feature_name1>
<feature_name2> ...
```
示例<br />
```
GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin
prototxt/c3d_sport1m_feature_extractor_frm.prototxt 
conv3d_deepnetA_sport1m_iter_1900000 0 50 1 
prototxt/output_list_prefix.txt fc7­1 fc6­1 prob
```
