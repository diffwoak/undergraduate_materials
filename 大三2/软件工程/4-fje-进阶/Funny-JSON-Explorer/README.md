## Funny JSON Explorer

##### 设计模式

（**迭代器模式**）collection读取文件得到集合，iterator将集合转化为node类对象并遍历，（**访问者模式**）visitor调用iterator对json中元素遍历，继承Visitor类实现不同风格（Tree/Rectangle）展示，在Visitor中定义多个方法设置图标风格（poker-face）

##### 类图

![image-20240616181701510](https://gitee.com/e-year/images/raw/master/img/202406161817086.png)



**添加新的风格**

添加style，需继承Visitor类到一个具体类，为其编写visit_all函数；在FunnyJsonExplorer类中加入条件判定使用

添加icon_family，需在Visitor父类中加入一个新的方法对应icon的设置；在Visitor类初始化中加入判定使用

##### 运行结果

![image-20240531205533102](https://gitee.com/e-year/images/raw/master/img/202406011042524.png)

![image-20240531205554702](https://gitee.com/e-year/images/raw/master/img/202406011042213.png)

![image-20240531205620441](https://gitee.com/e-year/images/raw/master/img/202406011043074.png)

![image-20240531205654980](https://gitee.com/e-year/images/raw/master/img/202406011043592.png)
