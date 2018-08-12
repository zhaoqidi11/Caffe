将该文件**右击**用**记事本**打开，另存为的时候选择编码为**unicode**即可<br />
>
该方法来自https://www.cnblogs.com/hydor/p/4914971.html
>
经过试验，以上方法无效（手动笑哭），会**无法编译**<br />
新方法<br />
>
来自https://social.msdn.microsoft.com/Forums/zh-CN/0fd13c75-f941-4b2f-b984-d7df2b2ab045/vs2015251712432025991202143838239064?forum=vstudiozhchs<br />
使用**notepad++**
打开乱码的后缀为py的文件<br />
在菜单中选择**格式**，然后在下拉菜单中选择**转为UTF-8无BOM编码格式**，然后保存<br />
>
再次使用vs2015打开即可
