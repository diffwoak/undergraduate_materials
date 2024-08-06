## Vue项目练习

之前已经下载过vue了，查看``vue -V``,结果版本太老2.9，卸载都卸载不掉参考[【已解决】卸载vue-cli过程中npm uninstall vue-cli -g 一直显示 up to date in 0.042s无法卸载。-CSDN博客](https://blog.csdn.net/qq_43055855/article/details/113124738?ops_request_misc=%7B%22request%5Fid%22%3A%22163169678216780271532702%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=163169678216780271532702&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-113124738.first_rank_v2_pc_rank_v29&utm_term=npm uninstall vue-cli -g 卸载失败&spm=1018.2226.3001.4187)解决

重新下载vue

```shell
# 最新淘宝镜像源
npm config set registry https://registry.npmmirror.com
npm install -g @vue/cli 
```



创建项目

```shell
vue create my-vue-app
vue create vue-21307347-chenxy
npm run serve
```

运行

![image-20240513120210987](C:\Users\asus\Desktop\大三下\软件工程\vue-proj\Vue项目练习.assets\image-20240513120210987.png)

![image-20240513120124447](C:\Users\asus\Desktop\大三下\软件工程\vue-proj\Vue项目练习.assets\image-20240513120124447.png)

打包并运行

```
npm run build
npm install http-server -g
cd dist
http-server
```

![image-20240513121047630](C:\Users\asus\Desktop\大三下\软件工程\vue-proj\Vue项目练习.assets\image-20240513121047630.png)

添加Markdown Loader 加载接口设计

JavaScript的注释无法显示图片，为了更好的可理解性，自己定制开发``markdown-loader``，配合vue脚手架，使其能够直接加载在设计文档(Markdown格式)中的源码。

在``高层设计.md``中，描述接口``class MarkdownLoaderExample``

```
class MarkdownLoaderExample{
getMsg(){
	throw Error('Must implemented by concrete class')
}
}
```

在``my-markdown-loader-example.js``中，形如下例，实现接口，然后在App.vue中使用

```
//引用Markdown文档中内嵌代码import MarkdownLoaderExample from './高层设计.md'

class MyMarkdownLoaderExample extends MarkdownLoaderExample{
...
getMsg(){
//返回自己特定的Msg
}
}
```



在vue.config.js中添加

```config
configureWebpack: {
    resolve: { fallback: { fs: false } }
  }
```



待解决：

markdown使用浏览器时无法使用fs读取markdown文件

如何读入markdown

[记一次 Vue 3 动态展示 markdown 文件 (desnlee.com)](https://desnlee.com/post/vue3-markdown/)

[在vue中解析md文档并显示-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1908057)

[Vue系列-动态引入markdown文件并显示的完整实现案例_vue-markdown-loader-CSDN博客](https://blog.csdn.net/ws6afa88/article/details/108700045)

### 参考链接

[01.课程介绍_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV12J411m7MG/?p=1)

[Home | Vue CLI (vuejs.org)](https://cli.vuejs.org/zh/#起步)

[简介 | Vue.js (vuejs.org)](https://cn.vuejs.org/guide/introduction.html)

安装

[VUE快速入门手册——安装环境Node.js - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/666409191)

[VUE快速入门手册——安装VUE脚手架（创建自己的第一个vue项目） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/666412575)

[VUE快速入门手册——VUE脚手架目录结构 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/666448165)

[国内npm源镜像（npm加速下载） 指定npm镜像_npm 国内镜像-CSDN博客](https://blog.csdn.net/qq_43940789/article/details/131449710)

使用

[VUE入门+5个小案例_vue案例-CSDN博客](https://blog.csdn.net/Beyondczn/article/details/113945908)

[03.Vue基础-第一个Vue程序_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV12J411m7MG?p=3&vd_source=6b153427393db22a222ff8aec7bb5efb)

打包运行

[vue项目打包步骤及运行打包项目_vue打包运行-CSDN博客](https://blog.csdn.net/WuqibuHuan/article/details/114823152)
