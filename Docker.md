# 概述

**(1)基本介绍**

Docker 是一个开源的应用容器引擎，基于 Go 语言 并遵从 Apache2.0 协议开源。

Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。

容器是完全使用沙箱机制，相互之间不会有任何接口（类似 iPhone 的 app）,更重要的是容器性能开销极低。

Docker 从 17.03 版本之后分为 CE（Community Edition: 社区版） 和 EE（Enterprise Edition: 企业版），我们用社区版就可以了。官网：https://docs.docker.com/

**(2)应用场景**

- Web 应用的自动化打包和发布。
- 自动化测试和持续集成、发布。
- 在服务型环境中部署和调整数据库或其他的后台应用。
- 从头编译或者扩展现有的 OpenShift 或 Cloud Foundry 平台来搭建自己的 PaaS 环境。

**(3)Docker 的优势**

Docker 是一个用于开发，交付和运行应用程序的开放平台。Docker 使您能够将应用程序与基础架构分开，从而可以快速交付软件。借助 Docker，您可以与管理应用程序相同的方式来管理基础架构。通过利用 Docker 的方法来快速交付，测试和部署代码，您可以大大减少编写代码和在生产环境中运行代码之间的延迟。

1、快速，一致地交付您的应用程序。Docker 允许开发人员使用您提供的应用程序或服务的本地容器在标准化环境中工作，从而简化了开发的生命周期。

容器非常适合持续集成和持续交付（CI / CD）工作流程，请考虑以下示例方案：

您的开发人员在本地编写代码，并使用 Docker 容器与同事共享他们的工作。
他们使用 Docker 将其应用程序推送到测试环境中，并执行自动或手动测试。
当开发人员发现错误时，他们可以在开发环境中对其进行修复，然后将其重新部署到测试环境中，以进行测试和验证。
测试完成后，将修补程序推送给生产环境，就像将更新的镜像推送到生产环境一样简单。

2、响应式部署和扩展
Docker 是基于容器的平台，允许高度可移植的工作负载。Docker 容器可以在开发人员的本机上，数据中心的物理或虚拟机上，云服务上或混合环境中运行。

Docker 的可移植性和轻量级的特性，还可以使您轻松地完成动态管理的工作负担，并根据业务需求指示，实时扩展或拆除应用程序和服务。

3、在同一硬件上运行更多工作负载
Docker 轻巧快速。它为基于虚拟机管理程序的虚拟机提供了可行、经济、高效的替代方案，因此您可以利用更多的计算能力来实现业务目标。Docker 非常适合于高密度环境以及中小型部署，而您可以用更少的资源做更多的事情。

# 容器化技术

虚拟化技术特点：1.资源占用多 2.冗余步骤多 3.启动很慢

容器化技术：容器化技术不是模拟的一个完整的操作系统

比较Docker和虚拟机的**区别**：
1.传统虚拟机，虚拟出硬件，运行一个完整的操作系统，然后在这个系统上安装和运行软件。

2.Docker容器内的应用直接运行在宿主机的内容，容器是没有自己的内核的，也没有虚拟硬件。

3.每个容器都是相互隔离的，每个容器都有属于自己的文件系统，互不影响。

容器化带来的好处：

![5WjNjq](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/5WjNjq.jpg)

# 基本组成

Docker的基本组成图如下：

![tVO72m](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/tVO72m.jpg)

说明：

![Gl9Oex](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/Gl9Oex.jpg)

# 运行流程

启动一个容器，Docker的运行流程如下图：

![IaYpFq](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/IaYpFq.jpg)

Docker是一个Client-Server结构的系统，Docker的守护进程运行在主机上，通过Socket从客户端访问！Docker Server接收到Docker-Client的指令，就会执行这个指令！

![xnqxye](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/xnqxye.jpg)

Docker为什么比VM Ware快？

1、Docker比虚拟机更少的抽象层

2、docker利用宿主机的内核，VM需要的是Guest OS

![IUzo10](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/IUzo10.jpg)

Docker新建一个容器的时候，不需要像虚拟机一样重新加载一个操作系统内核，直接利用宿主机的操作系统，而虚拟机是需要加载Guest OS。Docker和VM的对比如下：

![Ka42KZ](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/Ka42KZ.jpg)



# 常用命令

![jdjIOl](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/21/jdjIOl.jpg)

## 基础命令

```shell
docker version          #查看docker的版本信息
docker info             #查看docker的系统信息,包括镜像和容器的数量
docker 命令 --help       #帮助命令(可查看可选的参数)
docker COMMAND --help
```

## 镜像命令

### docker images

查看本地主机的所有镜像

```shell
➜  ~ docker images
REPOSITORY      TAG          IMAGE ID       CREATED        SIZE
arm64v8/mysql   8.0-oracle   55ca64bef8c5   6 days ago     482MB
hello-world     latest       46331d942d63   3 months ago   9.14kB

#解释:
1.REPOSITORY  镜像的仓库源

2.TAG  镜像的标签

3.IMAGE ID 镜像的id

4.CREATED 镜像的创建时间

5.SIZE 镜像的大小


# 可选参数

-a/--all 列出所有镜像

-q/--quiet 只显示镜像的id
```

### docker search

搜索镜像

```shell
➜  ~ docker search mysql
NAME                           DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
mysql                          MySQL is a widely used, open-source relation…   12755     [OK]
mariadb                        MariaDB Server is a high performing open sou…   4896      [OK]
percona                        Percona Server is a fork of the MySQL relati…   579       [OK]
phpmyadmin                     phpMyAdmin - A web interface for MySQL and M…   556       [OK]
bitnami/mysql                  Bitnami MySQL Docker Image                      71                   [OK]

#可选参数

Search the Docker Hub for images

Options:
  -f, --filter filter   Filter output based on conditions provided
      --format string   Pretty-print search using a Go template
      --limit int       Max number of search results (default 25)
      --no-trunc        Don't truncate output

#搜索收藏数大于3000的镜像
➜  ~ docker search mysql --filter=STARS=3000
NAME      DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
mysql     MySQL is a widely used, open-source relation…   12755     [OK]
mariadb   MariaDB Server is a high performing open sou…   4896      [OK]
```

### docker pull 镜像名[:tag] 

下载镜像

```shell
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker pull mysql
Using default tag: latest            #如果不写tag默认就是latest
latest: Pulling from library/mysql
6ec7b7d162b2: Pull complete          #分层下载,docker image的核心-联合文件系统
fedd960d3481: Pull complete
7ab947313861: Pull complete
64f92f19e638: Pull complete
3e80b17bff96: Pull complete
014e976799f9: Pull complete
59ae84fee1b3: Pull complete
ffe10de703ea: Pull complete
657af6d90c83: Pull complete
98bfb480322c: Pull complete
6aa3859c4789: Pull complete
1ed875d851ef: Pull complete
Digest: sha256:78800e6d3f1b230e35275145e657b82c3fb02a27b2d8e76aac2f5e90c1c30873 #签名
Status: Downloaded newer image for mysql:latest
docker.io/library/mysql:latest  #下载来源的真实地址  #docker pull mysql等价于docker pull docker.io/library/mysql:latest

```

指定版本下载

```shell
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker pull mysql:5.7
5.7: Pulling from library/mysql
6ec7b7d162b2: Already exists
fedd960d3481: Already exists
7ab947313861: Already exists
64f92f19e638: Already exists
3e80b17bff96: Already exists
014e976799f9: Already exists
59ae84fee1b3: Already exists
7d1da2a18e2e: Pull complete
301a28b700b9: Pull complete
529dc8dbeaf3: Pull complete
bc9d021dc13f: Pull complete
Digest: sha256:c3a567d3e3ad8b05dfce401ed08f0f6bf3f3b64cc17694979d5f2e5d78e10173
Status: Downloaded newer image for mysql:5.7
docker.io/library/mysql:5.7
```

### docker rmi

删除镜像

```shell
#1.删除指定的镜像id
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker rmi -f  镜像id
#2.删除多个镜像id
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker rmi -f  镜像id 镜像id 镜像id
#3.删除全部的镜像id
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker rmi -f  $(docker images -aq)
```

## 容器命令

拉取一个centos镜像

```shell
docker pull centos
```

运行容器的命令说明：

```shell
docker run [可选参数] image

#参数说明
--name="名字"           指定容器名字
-d                     后台方式运行
-it                    使用交互方式运行,进入容器查看内容
-p                     指定容器的端口
(
-p ip:主机端口:容器端口  配置主机端口映射到容器端口
-p 主机端口:容器端口
-p 容器端口
)
-P                     随机指定端口(大写的P)
# docker -port 容器ID/容器名称
```

运行并进入容器centos

```shell
➜  ~ docker run -it centos
[root@579e759697ff /]# ls
bin  etc   lib	  lost+found  mnt  proc  run   srv  tmp  var
dev  home  lib64  media       opt  root  sbin  sys  usr
```

退出容器命令：

```shell
#exit 停止并退出容器（后台方式运行则仅退出）
#Ctrl+P+Q  不停止容器退出
[root@56cc94635433 /]# exit
exit
➜  ~
```

列出运行过的容器命令：

```shell
#docker ps 
     # 列出当前正在运行的容器
-a   # 列出所有容器的运行记录
-n=? # 显示最近创建的n个容器
-q   # 只显示容器的编号

➜  ~ docker ps
CONTAINER ID   IMAGE     COMMAND       CREATED              STATUS              PORTS     NAMES
579e759697ff   centos    "/bin/bash"   About a minute ago   Up About a minute             elegant_engelbart
➜  ~ docker ps -a
CONTAINER ID   IMAGE         COMMAND       CREATED              STATUS                      PORTS     NAMES
56cc94635433   centos        "/bin/bash"   About a minute ago   Exited (0) 59 seconds ago             unruffled_jemison
6e485006e05c   hello-world   "/hello"      13 hours ago         Exited (0) 13 hours ago               vibrant_mayer
```

删除容器命令：

```shell
docker rm 容器id                 #删除指定的容器,不能删除正在运行的容器,强制删除使用 rm -f
docker rm -f $(docker ps -aq)   #删除所有的容器
docker ps -a -q|xargs docker rm #删除所有的容器
```

启动和停止容器命令：

```shell
docker start 容器id          #启动容器
docker restart 容器id        #重启容器
docker stop 容器id           #停止当前运行的容器
docker kill 容器id           #强制停止当前容器
```

导出容器

```shell
# 导出容器 1e560fca3906 快照到本地文件 ubuntu.tar。
$ docker export 1e560fca3906 > ubuntu.tar
```

使用 docker import 从容器快照文件中再导入为镜像

```shell
$ cat docker/ubuntu.tar | docker import - test/ubuntu:v1
# 也可以使用URL导入
$ docker import http://example.com/exampleimage.tgz example/imagerepo
```

## Dockerfile

Dockerfile 是一个用来构建镜像的文本文件，文本内容包含了一条条构建镜像所需的指令和说明。

**使用Dockerfile定制镜像**

在一个空目录下，新建一个名为`Dockerfile`文件，并在文件内添加以下内容：

```shell
# 定制的镜像都是基于 FROM 的镜像，这里的 nginx 就是定制需要的基础镜像。后续的操作都是基于 nginx。
FROM nginx
# 用于执行后面跟着的命令行命令。
RUN echo '这是一个本地构建的nginx镜像' > /usr/share/nginx/html/index.html
```

**注意：**Dockerfile 的指令每执行一次都会在 docker 上新建一层。所以过多无意义的层，会造成镜像膨胀过大。例如：

```shell
FROM centos
# 这会创建三层镜像
RUN yum -y install wget
RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz"
RUN tar -xvf redis.tar.gz

# 可简化为以下格式
# 命令间以 && 符号连接，这样只会创建1层镜像
FROM centos
RUN yum -y install wget \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
    && tar -xvf redis.tar.gz
```

**开始构建镜像**

通过目录下的Dockerfile构建一个nginx:v3（镜像名称：镜像标签）

```shell
# 最后的.代表本次执行的上下文路径
$ docker build -t nginx:v3 .
```

![image-20220926144140435](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/09/26/image-20220926144140435.png)



## 其他常用命令

### 日志的查看

```shell
➜  ~ docker logs --help

Usage:  docker logs [OPTIONS] CONTAINER

Fetch the logs of a container

Options:
      --details        Show extra details provided to logs
  -f, --follow         Follow log output
      --since string   Show logs since timestamp (e.g. 2013-01-02T13:23:37Z) or relative (e.g.
                       42m for 42 minutes)
  -n, --tail string    Number of lines to show from the end of the logs (default "all")
  -t, --timestamps     Show timestamps
      --until string   Show logs before a timestamp (e.g. 2013-01-02T13:23:37Z) or relative
                       (e.g. 42m for 42 minutes)

# 常用：
docker logs -tf 容器id
docker logs --tail number 容器id #num为要显示的日志条数

# docker容器后台运行，必须要有一个前台的进程，否则会自动停止
# 编写shell脚本循环执行，使得centos容器保持运行状态
➜  ~ docker run -d centos /bin/sh  -c "while true;do echo hi;sleep 5;done"
c125e3e6351fa51f8ceb4afccfab20c89e7d49deba9b487733b1899bf0c61001
➜  ~ docker ps
CONTAINER ID   IMAGE     COMMAND                  CREATED         STATUS         PORTS     NAMES
c125e3e6351f   centos    "/bin/sh -c 'while t…"   6 seconds ago   Up 5 seconds             dazzling_haslett
➜  ~ docker logs -tf --tail 10 c125e3e6351f
2022-06-21T15:08:00.914156250Z hi
2022-06-21T15:08:05.925762919Z hi
2022-06-21T15:08:10.943662922Z hi
2022-06-21T15:08:15.956591133Z hi
2022-06-21T15:08:20.968613093Z hi
2022-06-21T15:08:25.989417721Z hi
2022-06-21T15:08:31.006225250Z hi
2022-06-21T15:08:36.021680545Z hi
2022-06-21T15:08:41.030521005Z hi
2022-06-21T15:08:46.042694674Z hi
2022-06-21T15:08:51.063882510Z hi
2022-06-21T15:08:56.071267679Z hi
2022-06-21T15:09:01.096717209Z hi
```

### 查看容器中进程信息

```shell
➜  ~ docker top c125e3e6351f
UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
root                3026                3005                0                   15:06               ?                   00:00:00            /bin/sh -c while true;do echo hi;sleep 5;done
root                3119                3026                0                   15:11               ?                   00:00:00            /usr/bin/coreutils --coreutils-prog-shebang=sleep /usr/bin/sleep 5
```

### 查看容器的元数据

```shell
docker inspect 容器id
```

### 进入当前正在运行的容器

因为通常我们的容器都是使用后台方式来运行的，有时需要进入容器修改配置

方式一：

```shell
➜  ~ docker exec -it c125e3e6351f /bin/bash
[root@c125e3e6351f /]# ls
bin  etc   lib	  lost+found  mnt  proc  run   srv  tmp  var
dev  home  lib64  media       opt  root  sbin  sys  usr
[root@c125e3e6351f /]# ps -ef
UID        PID  PPID  C STIME TTY          TIME CMD
root         1     0  0 15:06 ?        00:00:00 /bin/sh -c while true;do echo hi;sleep 5;done
root       142     0  0 15:17 pts/0    00:00:00 /bin/bash
root       160     1  0 15:18 ?        00:00:00 /usr/bin/coreutils --coreutils-prog-shebang=sleep /usr
root       161   142  0 15:18 pts/0    00:00:00 ps -ef
```

方式二：

```shell
➜  ~ docker attach c125e3e6351f
hi
hi
```

>docker exec 进入容器后开启一个新的终端，可以在里面操作，推荐使用，退出不会导致容器结束
>
>docker attach 进入容器正在执行的终端，不会启动新的进程
>

### 拷贝操作

拷贝操作的命令如下：

```shell
#拷贝容器的文件到主机中
docker cp 容器id:容器内路径  目的主机路径

#拷贝宿主机的文件到容器中
docker cp 目的主机路径 容器id:容器内路径
```

```shell
➜  ~ docker exec -it c125e3e6351f /bin/bash
[root@c125e3e6351f /]# ls
bin  etc   lib	  lost+found  mnt  proc  run   srv  tmp  var
dev  home  lib64  media       opt  root  sbin  sys  usr
# 新建文件template.java
[root@c125e3e6351f /]# cd home
[root@c125e3e6351f home]# ls
[root@c125e3e6351f home]# touch template.java
[root@c125e3e6351f home]# ls
template.java
[root@c125e3e6351f home]# read escape sequence
# 拷贝
➜  ~ docker cp c125e3e6351f:/home/template.java /Users/eumenides/Desktop/
➜  ~ ls ./Desktop/
Docker.md     template.java 题库管理
```

# Docker镜像

## 什么是镜像

镜像是一种轻量级、可执行的独立软件包，用来打包软件运行环境和基于运行环境开发的软件，它包含运行某个软件所需要的所有内容，包括代码，运行时（一个程序在运行或者在被执行的依赖）、库，环境变量和配置文件。

## 镜像加载原理

Docker的镜像实际上由一层一层的文件系统组成，这种层级的文件系统是UnionFS联合文件系统。

UnionFS (联合文件系统) ：Union文件系统( UnionFS)是一种分层、轻量级并且高性能的文件系统，它支持对文件系统的修改作为一次提交来一层层的叠加，同时可以将不同目录挂载到同一个虚拟文件系统下(unite several directories into a single virtual filesystem)。Union 文件系统是Docker镜像的基础。镜像可以通过分层来进行继承,基于基础镜像(没有父镜像) , 可以制作各种具体的应用镜像。

特性：一次同时加载多个文件系统,但从外面看起来,只能看到一个文件系统,联合加载会把各层文件系统叠加起来,这样最终的文件系统会包含所有底层的文件和目录

bootfs(boot file system)主要包含bootloader和kernel, bootloader主要是引导加载kernel，Linux刚启动时会加载bootfs文件系统，在Docker镜像的最底层是bootfs。这一层与我们典型的Linux/Unix系统是一样的 ,包含boot加载器和内核。当boot加载完成之后整个内核就都在内存中了,此时内存的使用权已由bootfs转交给内核,此时系统也会卸载bootfs.

rootfs (root file system) ，在bootfs之上。包含的就是典型Linux系统中的/dev, /proc, /bin, /etc等标准目录和文件。rootfs就是各种不同的操作系统发行版,比如Ubuntu，Centos等等。

## 分层理解

下载镜像时，注意观察下载时的日志输出，可以看到是一层一层在下载的！

![b2aaNi](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/08/07/b2aaNi.png)

思考：为什么Docker镜像要采用这种分层的结构呢？

最大的好处就是资源共享！比如有多个镜像都从相同的base镜像构建而来，那么宿主机只需在磁盘上保存一份base镜像，同时内存中也只需要加载一份base镜像，这样就可以为所有的容器服务了，而且镜像的每一层都可以被共享。

查看镜像分层的方式可以通过`docker image inspect`命令

```
"RootFS": {
"Type": "layers",
"Layers": [
"sha256:e4f85c7c2b1fddac675cb2be9672426c32293e7a509e2c2e6a198394aa46bbd1",
"sha256:8cd8ecc954724d1c59a15aa17b2d2c368e9016d6a8b28d0df6dc545b0b914130",
"sha256:b6d8674d26d7abe1774f4e180ddd8f9b8771d85602fba88dddc6108827b65ec1",
"sha256:4dcd470a828ef3ebed21049ba2e311ccbba8c8b7cd12b8cf46c31b8741cb127e",
"sha256:74e29ed0c55e882d28afa1127a6dd286c67bc61ffc4538e20c17d5ff51245697",
"sha256:0921b7818544182d665d44f9158dd07a85f65622de8cd41222d42ebc1ddb373a"
]
},
```

![uyk6RD](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/08/07/uyk6RD.jpg)

![heugqN](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/08/07/heugqN.jpg)

![xnWYHP](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/08/07/xnWYHP.jpg)

## 提交镜像

> 使用docker commit 命令提交容器成为一个新的版本
>
> docker commit -m=“提交的描述信息”  -a="作者" 容器id 目标镜像名:[TAG] 

由于默认的Tomcat镜像的webapps文件夹中没有任何内容，需要从webapps.dist中拷贝文件到webapps文件夹。下面自行制作镜像：就是从webapps.dist中拷贝文件到webapps文件夹下，并提交该镜像作为一个新的镜像。使得该镜像默认的webapps文件夹下就有文件。具体命令如下：

```shell
#1.复制文件夹
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker run -it tomcat /bin/bash
root@2a3bf3eaa2e4:/usr/local/tomcat# cd webapps
root@2a3bf3eaa2e4:/usr/local/tomcat/webapps# ls
root@2a3bf3eaa2e4:/usr/local/tomcat/webapps# cd ../
root@2a3bf3eaa2e4:/usr/local/tomcat# cp -r webapps.dist/* webapps
root@2a3bf3eaa2e4:/usr/local/tomcat# cd webapps
root@2a3bf3eaa2e4:/usr/local/tomcat/webapps# ls
ROOT  docs  examples  host-manager  manager
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker ps
CONTAINER ID   IMAGE                 COMMAND        CREATED         STATUS         PORTS                    NAMES
2a3bf3eaa2e4   tomcat                "/bin/bash"    4 minutes ago   Up 4 minutes   8080/tcp                 competent_torvalds
7789d4505a00   portainer/portainer   "/portainer"   24 hours ago    Up 24 hours    0.0.0.0:8088->9000/tcp   quirky_sinoussi
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker exec -it 2a3bf3eaa2e4 /bin/bash
root@2a3bf3eaa2e4:/usr/local/tomcat# cd webapps
root@2a3bf3eaa2e4:/usr/local/tomcat/webapps# ls
ROOT  docs  examples  host-manager  manager
root@2a3bf3eaa2e4:/usr/local/tomcat/webapps# cd ../
root@2a3bf3eaa2e4:/usr/local/tomcat# read escape sequence
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker ps
CONTAINER ID   IMAGE                 COMMAND        CREATED         STATUS         PORTS                    NAMES
2a3bf3eaa2e4   tomcat                "/bin/bash"    8 minutes ago   Up 8 minutes   8080/tcp                 competent_torvalds
7789d4505a00   portainer/portainer   "/portainer"   24 hours ago    Up 24 hours    0.0.0.0:8088->9000/tcp   quirky_sinoussi

#2.提交镜像作为一个新的镜像
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker commit -m="add webapps" -a="Ethan" 2a3bf3eaa2e4 mytomcat:1.0
sha256:f189aac861de51087af5bc88a5f1de02d9574e7ee2d163c647dd7503a2d3982b
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker images
REPOSITORY            TAG       IMAGE ID       CREATED         SIZE
mytomcat              1.0       f189aac861de   7 seconds ago   653MB
mysql                 5.7       f07dfa83b528   6 days ago      448MB
tomcat                latest    feba8d001e3f   10 days ago     649MB
nginx                 latest    ae2feff98a0c   12 days ago     133MB
centos                latest    300e315adb2f   2 weeks ago     209MB
portainer/portainer   latest    62771b0b9b09   5 months ago    79.1MB
elasticsearch         7.6.2     f29a1ee41030   9 months ago    791MB

#3.运行容器
[root@iZwz99sm8v95sckz8bd2c4Z ~]# docker run -it mytomcat:1.0 /bin/bash
root@1645774d4605:/usr/local/tomcat# cd webapps
root@1645774d4605:/usr/local/tomcat/webapps# ls
ROOT  docs  examples  host-manager  manager
wz99sm8v95sckz8bd2c4Z ~]# docker images
REPOSITORY            TAG       IMAGE ID       CREATED         SIZE
mytomcat              1.0       f189aac861de   7 seconds ago   653MB
mysql                 5.7       f07dfa83b528   6 days ago      448MB
tomcat                latest    feba8d001e3f   10 days ago     649MB
nginx                 latest    ae2feff98a0c   12 days ago     133MB
centos                latest    300e315adb2f   2 weeks ago     209MB
portainer/portainer   latest    62771b0b9b09   5 months ago    79.1MB
elasticsearch         7.6.2     f29a1ee41030   9 months ago    791MB

```



# 安装PHP
**获取镜像**

```shell
docker pull php
```

**启动PHP**

```shell
docker run --name  myphp -v ~/nginx/www:/www  -d php
```

命令说明：

- **--name myphp** : 将容器命名为 myphp。
- **-v ~/nginx/www:/www** : 将主机中项目的目录 www 挂载到容器的 /www

**创建 ~/nginx/conf/conf.d 目录：**

```shell
mkdir ~/nginx/conf/conf.d 
```

**在该目录下添加 ~/nginx/conf/conf.d/runoob-test-php.conf 文件，内容如下：**

```nginx
server {
    listen       80;
    server_name  localhost;

    location / {
        root   /usr/share/nginx/html;
        index  index.html index.htm index.php;
    }

    error_page   500 502 503 504  /50x.html;
    location = /50x.html {
        root   /usr/share/nginx/html;
    }

    location ~ \.php$ {
        fastcgi_pass   php:9000;
        fastcgi_index  index.php;
        fastcgi_param  SCRIPT_FILENAME  /www/$fastcgi_script_name;
        include        fastcgi_params;
    }
}
```

配置文件说明：

- **php:9000**: 表示 php 服务的 URL，下面我们会具体说明。
- **/www/**: 是 **myphp** 中 php 文件的存储路径，映射到本地的 ~/nginx/www 目录。

**启动 nginx：**

```shell
docker run --name runoob-php-nginx -p 8083:80 -d \
    -v ~/nginx/www:/usr/share/nginx/html:ro \
    -v ~/nginx/conf/conf.d:/etc/nginx/conf.d:ro \
    --link myphp:php \
    nginx
```

- **-p 8083:80**: 端口映射，把 **nginx** 中的 80 映射到本地的 8083 端口。
- **~/nginx/www**: 是本地 html 文件的存储目录，/usr/share/nginx/html 是容器内 html 文件的存储目录。
- **~/nginx/conf/conf.d**: 是本地 nginx 配置文件的存储目录，/etc/nginx/conf.d 是容器内 nginx 配置文件的存储目录。
- **--link myphp:php**: 把 **myphp** 的网络并入 ***nginx\***，并通过修改 **nginx** 的 /etc/hosts，把域名 **php** 映射成 127.0.0.1，让 nginx 通过 php:9000 访问 php-fpm。

**接下来我们在 ~/nginx/www 目录下创建 index.php，代码如下：**

```php
<?php
echo phpinfo();
?>
```

**浏览器打开 http://127.0.0.1:8083/index.php，显示如下：**

![img](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/09/26/4CA3D4DE-3883-449C-B2F2-7C80D9A5B384.jpg)

