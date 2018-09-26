感觉可能会是比较常犯的一个错误，记录一下<br />
载入模型定义文件的时候出现问题：<br />
[libprotobuf ERROR C:\Users\guillaume\work\caffe-builder\build_v140_x62\packages\protobuf\protobuf_download-prefix\src\protobuf_download\src\google\protobuf\text_format.cc:298]Error parsing text-format caffe.NetParameter:86xxx:
Invalid escape sequence in string literal.
错误原因：输入层的source：地址中出现了'\'，似乎只能用'/'或者'\\'代替……
