# Funny JSON Explorer

Funny JSON Explorer（**FJE**），是一个JSON文件可视化的命令行界面小工具

```shell
fje -f <json file> -s <style> -i <icon family>
```

```
{
    oranges: {
        'mandarin': {                            ├─ oranges
            clementine: null,                    │  └─ mandarin
            tangerine: 'cheap & juicy!'  -=>     │     ├─ clementine
        }                                        │     └─ tangerine: cheap & juicy!
    },                                           └─ apples
    apples: {                                       ├─ gala
        'gala': null,                               └─ pink lady
        'pink lady': null
    }
}
````

FJE可以快速切换**风格**（style），包括：树形（tree）、矩形（rectangle）；

```

├─ oranges                             ┌─ oranges ───────────────────────────────┐
│  └─ mandarin                         │  ├─ mandarin ───────────────────────────┤
│     ├─ clementine                    │  │  ├─ clementine ──────────────────────┤
│     └─ tangerine: cheap & juicy!     │  │  ├─ tangerine: cheap & juicy! ───────┤
└─ apples                              ├─ apples ────────────────────────────────┤
   └─ gala                             └──┴─ gala ───────────────────────────────┘

        树形（tree）                                   矩形（rectangle）
````

也可以指定**图标族**（icon family），为中间节点或叶节点指定一套icon

```
├─♢oranges                                 
│  └─♢mandarin                             
│     ├─♤clementine                        
│     └─♤tangerine: cheap & juicy!    
└─♢apples                                  
   └─♤gala                                 

poker-face-icon-family: 中间节点icon：♢ 叶节点icon：♤                 
```



## 领域模型 

![domain-model](Funny JSON Explorer 领域模型.png)



## 作业要求

基于上述需求描述和领域模型，按照设计模式要求，进行软件设计，并编码实现（任何语言均可）。

### 设计模式
使用**工厂方法**（Factory）、**抽象工厂**（Abstract Factory）、**建造者**（Builder）模式、**组合模式**（Composition），完成功能的同时，使得程序易于扩展和维护。
1. （必做）：不改变现有代码，只需添加新的抽象工厂，即可添加新的风格
2. （选做）：通过配置文件，可添加新的图标族

### 作业提交
1. 设计文档：类图与说明，说明使用的设计模式及作用
2. 运行截图：两种风格，两种图标族，共计4次运行fje的屏幕截图
3. 源代码库：公开可访问的Github repo URL



### 思路

使用抽象工厂设计模式，能够创建相关或依赖对象的家族，不需要指定具体的类，可派生多个具体工厂类和产品类，一个工厂实例能够创建多个该工厂对应的产品实例，这里对应了领域模型中Container和Leaf的关系。将Container根据不同的style抽象成不同的具体类，再定义另一种IconFactory工厂类，抽象不同icon family类型。





![image-20240531203911405](https://gitee.com/e-year/images/raw/master/img/202406011042628.png)



添加新的风格：添加style，仅需继承Container类到一个具体类，为其编写draw函数；添加icon_family仅需继承IconFactory类和IconProduct类成一个新的icon_family类。在FunnyJsonExplorer中加入条件判断调用新的类即可

![image-20240531205533102](https://gitee.com/e-year/images/raw/master/img/202406011042524.png)

![image-20240531205554702](https://gitee.com/e-year/images/raw/master/img/202406011042213.png)

![image-20240531205620441](https://gitee.com/e-year/images/raw/master/img/202406011043074.png)

![image-20240531205654980](https://gitee.com/e-year/images/raw/master/img/202406011043592.png)

还差：

1. 说明设计模式
2. github推送

[Repository search results (github.com)](https://github.com/search?q=Funny JSON Explorer&type=repositories)

[设计模式之-3种常见的工厂模式简单工厂模式、工厂方法模式和抽象工厂模式，每一种模式的概念、使用场景和优缺点。_工厂方法 抽象工厂 代码-CSDN博客](https://blog.csdn.net/qq_42262444/article/details/135160468?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-135160468-blog-80342594.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

[Java设计模式：工厂模式——图文+代码示例（通俗易懂）-CSDN博客](https://blog.csdn.net/A_hxy/article/details/105758657)

[【深入设计模式】建造者模式—带你彻底弄懂建造者模式_解析真实建造者模式-CSDN博客](https://blog.csdn.net/qq_38550836/article/details/125862850)

[【设计模式】组合模式 ( 简介 | 适用场景 | 优缺点 | 代码示例 )-CSDN博客](https://blog.csdn.net/shulianghan/article/details/119822802)

## 参考资料

1. unicode 制表符与图标： https://unicode.yunser.com/