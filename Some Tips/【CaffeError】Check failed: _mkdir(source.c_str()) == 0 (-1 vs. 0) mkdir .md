在使用create_lmdb.sh的过程中出现了问题<br />
如下所示：<br />
```
Creating train lmdb right...
I1011 10:19:53.109196 25332 convert_imageset.cpp:89] A total of 9693 images.
F1011 10:19:53.110196 25332 db_lmdb.cpp:18] Check failed: _mkdir(source.c_str()) == 0 (-1 vs. 0) mkdir E:\Meisa_SiameseNetwork\RAIDataset\imgDataset\siamese_train_right_lmdb failed
*** Check failure stack trace: ***

```
原因：要创建的文件夹已经存在，删除即可。
