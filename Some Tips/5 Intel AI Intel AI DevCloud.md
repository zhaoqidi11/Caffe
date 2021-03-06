### 1.在集群上计算
当您登录时，您将发现自己位于主机c009上，这是您的登录节点。<br />
此节点仅用于代码开发和编译，但不用于计算。<br />
这是因为它没有太多的计算能力，而且登录节点上的CPU时间和RAM使用也有限制;如果您的工作负载超过限制，您的工作负载将被杀死。<br />
要在强大的计算节点上运行计算工作负载，必须使用qsub指令通过Torque作业队列提交作业。有关示例作业脚本，请参阅第2节。<br />
您可以在https://access.colfaxresearch.com/<br />
(如果链接不工作，转到原来的欢迎电子邮件，然后点击指示链接。然后转到“计算”页面。<br />
<br />
### 2.基本作业提交
提交作业可以通过作业脚本文件完成。<br />
假设您有一个Python应用程序“my_application.py”。<br />
在同一个文件夹中，使用您最喜欢的文本编辑器并创建一个“myjob”文件。然后添加以下三行。<br />
```
#PBS -l nodes=1:ppn=2

   cd $PBS_O_WORKDIR
   
python my_application.py
```
第一行是一个特殊的命令，它请求一个Intel Xeon处理器和节点上的所有处理槽(请参阅access页面)。<br />
第二行确保脚本运行在与提交脚本的目录相同的目录中。第三行运行Python应用程序。<br />
你可以提交这份工作与:<br />
```
[u100@c001 ~]# qsub myjob
```
该命令将返回作业ID，这是作业的跟踪号。<br />
你可以通过以下方式来追踪这份工作:<br />
```
[u100@c001 ~]# qstat
```
一旦完成，输出将在文件中:<br />
```
[u100@c001 ~]# cat myjob.oXXXXXX

   [u100@c001 ~]# cat myjob.eXXXXXX
```

这里“XXXXXX”是作业ID， .o文件包含标准输出流，.e文件包含错误流。<br />
有关工作脚本的更多信息，请参见:<br />
https://access.colfaxresearch.com/?p=compute
##########################################################
### 3.运行多个作业

通过该集群，您可以访问80多个Intel Xeon可扩展处理器。 总之，您的理论峰值超过200 TFLOP / s的机器学习性能。<br />

但是，要获得此性能，您需要正确使用本节中讨论的群集。<br />

对于大多数机器学习工作负载，每个作业保留1个节点（这是默认值）。 <br />
如果您保留更多节点，则除非您明确使用分布式培训库（如MLSL），否则您的应用程序将无法利用它们。 <br />
大多数人没有。 保留额外节点，无论您的应用程序是否使用它们，都会降低未来作业的队列优先级。<br />
相反，要利用可用的多个节点，请提交具有不同参数的多个单节点作业。 例如，您可以提交具有不同学习率值的多个作业，如下所示：<br />
例如，您可以提交具有不同学习率值的多个作业，如下所示：<br />
*您的应用程序“my_application.py”应使用命令行参数来设置学习速率：<br />



