在平台上遇到的问题，经过一番折腾，解决方法如下（来自https://www.cnblogs.com/EasonJim/p/7487413.html）:<br />
```
sudo apt-get isntall unrar
```
解压命令
```
unrar x 你的压缩包名字.rar
```

关于7z文件<br />
来源：https://www.cnblogs.com/yiwd/p/3649094.html
在linux中输入<br />
```
sudo apt-get install p7zip-full
```
解压命令
```
7za x 文件名称 -r -o./
```
例如
```
7za x phpMyAdmin-3.3.8.1-all-languages.7z -r -o./
```
x  代表解压缩文件，并且是按原始目录树解压（还有个参数 e 也是解压缩文件，但其会将所有文件都解压到根下，而不是自己原有的文件夹下）<br />

phpMyAdmin-3.3.8.1-all-languages.7z  是压缩文件，这里我用phpadmin做测试。这里默认使用当前目录下的phpMyAdmin-3.3.8.1-all-languages.7z<br />

-r 表示递归解压缩所有的子文件夹<br />

-o 是指定解压到的目录，-o后是没有空格的，直接接目录。这一点需要注意。<br />
