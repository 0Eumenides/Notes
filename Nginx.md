# 概念

Nginx (engine x) 是一个高性能的HTTP和反向代理web服务器，同时也提供了IMAP/POP3/SMTP服务。Nginx是由伊戈尔·赛索耶夫为俄罗斯访问量第二的Rambler.ru站点（俄文：Рамблер）开发的，第一个公开版本0.1.0发布于2004年10月4日。2011年6月1日，nginx 1.0.4发布。

其特点是占有内存少，并发能力强，事实上nginx的并发能力在同类型的网页服务器中表现较好，中国大陆使用nginx网站用户有：百度、京东、新浪、网易、腾讯、淘宝等。在全球活跃的网站中有12.18%的使用比率，大约为2220万个网站。

Nginx 是一个安装非常的简单、配置文件非常简洁（还能够支持perl语法）、Bug非常少的服务。Nginx 启动特别容易，并且几乎可以做到7*24不间断运行，即使运行数个月也不需要重新启动。你还能够不间断服务的情况下进行软件版本的升级。

Nginx代码完全用C语言从头写成。官方数据测试表明能够支持高达 50,000 个并发连接数的响应。

# 作用

## 反向代理

> Http代理，反向代理：作为web服务器最常用的功能之一，尤其是反向代理。

正向代理

代理服务器代理客户端的请求，常见的应用有VPN。

![82ybBZ](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/82ybBZ.jpg)

反向代理

![lZALRC](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/lZALRC.jpg)

## 负载均衡

> Nginx提供的负载均衡策略有2种：内置策略和扩展策略。内置策略为轮询，加权轮询，Ip hash。扩展策略，就天马行空，只有你想不到的没有他做不到的。

轮询

![X6Z37D](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/X6Z37D.jpg)

加权轮询

![DtNUlv](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/DtNUlv.jpg)

iphash对客户端请求的ip进行hash操作，然后根据hash结果将同一个客户端ip的请求分发给同一台服务器进行处理，可以解决session不共享的问题。

![P7HAp3](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/P7HAp3.jpg)

## 动静分离

> 动静分离，在我们的软件开发中，有些请求是需要后台处理的，有些请求是不需要经过后台处理的（如：css、html、jpg、js等等文件），这些不需要经过后台处理的文件称为静态文件。让动态网站里的动态网页根据一定规则把不变的资源和经常变的资源区分开来，动静资源做好了拆分以后，我们就可以根据静态资源的特点将其做缓存操作。提高资源响应的速度。

![IrrsKz](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/25/IrrsKz.jpg)

目前，通过使用Nginx大大提高了我们网站的响应速度，优化了用户体验，让网站的健壮性更上一层楼！

# 安装

这里在docker中安装使用

下载镜像

```bash
docker pull nginx
```

启动nginx

```bash
docker run -d --name nginx01 -p 3334:80 nginx

-d 后台运行
--name 给容器命名
-p 3334:80 将宿主机的端口3334映射到该容器的80端口
```

访问nginx

```
http://localhost:3344/
```

![P87iv5](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/06/26/P87iv5.png)

# 常用命令

```bash
cd /usr/local/nginx/sbin/
./nginx  启动
./nginx -s stop  停止
./nginx -s quit  安全退出
./nginx -s reload  重新加载配置文件
ps aux|grep nginx  查看nginx进程
```

# 配置文件

```nginx
http {
    ...
	upstream cvzhanshi{    # 配置负载均衡
		server 127.0.0.1:8082/ weight=1;  # 权重
		server 127.0.0.1:8081/ weight=1;
	}


    server {
        listen       80;
        server_name  localhost;

        location / {
            root   html;
            index  index.html index.htm;
			proxy_pass http://cvzhanshi;   # 80端口
      															# 反向代理
        }
        ...
}

```

