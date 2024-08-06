### 要求

- 面向对象
- 符合Java编码规范[Code Conventions for the Java Programming Language: Contents (oracle.com)](https://www.oracle.com/java/technologies/javase/codeconventions-contents.html)
- 使用JDK附带文档工具javadoc，根据程序文档化注释，生成说明性文档

### 思路

- 看leetcode一些题，了解题解的技巧
- 看菜鸟教程
- 注意：有空解决一下warning的问题
- 有空看一下java的就业需求

### 功能

1. 用户输入薪金：计算个人所得税额 calculateTax
2. 系统：调整个人所得税起征点 
3. 系统：调整个人所得税各级税率

### 提交

- 源代码
- 设计文档 design.pdf（javadoc）
- 编译运行程序的脚本 .bat文件
- readme.txt纯文本的自述文件，描述姓名、学号等基本信息，与提交代码文件的简要解释
- 测试用例（尽量支持回归测试）

### 其他

- 测试用例：有助于提高正确性的措施（单元测试、回归测试），如使用单元测试框架JUnit作为测试工具
- 对异常情况处理，如nextInt()中输入小数报错的情况
- 用户界面是否友好
- 对未来变化的适应性，如是否容易修改应对可能发生的变化，是否方便在不同环境（如基于J2EE的Web应用中）复用程序或程序中的构件

### Java项目结构

1. ```
   src/	# 放Java源文件
   	com/
   		example/
   			project/
   				MyClass.java
   lib/	# 放外部依赖库
   	commons-lang.jar
   	junit.jar
   test/	# 放项目的单元测试代码
   	com/
   		example/
   			project/
   				MyClassTest.java
   resources/	#放项目的配置文件和其他资源文件，这些文件不需要编译，直接被复制到输出目录中
   	config.properties
   	log4j.xml
   	images/
   		logo.png
   docs/
   	requirements.md
   	design.md
   	api/
   		index.html
   ```

2. ```
   src：（source）存放所有资源和代码
           main：主程序
                   ○ java（源 根）：.java文件。
                   ○ resources（资源 根）：资源文件，如xml、properties配置文件。
                           templates：动态页面，如 thymeleaf 页面。
                                   需要服务器渲染，所以直接访问是无意义的，也访问不到。
                           static：静态资源，如 html、css、js、image。
                                   □ 可直接 localhost:8080/hello.html 访问该目录文件。
                                   □ 也可 return "hello.html"; 跳转。
                           编译后，resources和源根在同一目录下！
                   ○ lib：存放 jar包，需要设置添加到库。
                   ○ webapp：web资源
                           页面静态资源：html、css、js、图片 
                           WEB-INF：固定写法。此目录下的文件不能被外部(浏览器)直接访问。
                                   lib：jar包存放的目录
                                   web.xml：web 项目的配置文件(3.0规范之后可以省略)
                                   classes：target中，java编译生成class文件和资源文件存放的路径。对于war项目，配置文件中的classpath就是指这里。
           test：测试程序
                   java（测试 根）：.java文件。
                   resources（测试资源 根）：资源文件，如xml、properties配置文件。
   pom.xml：maven 配置文件。
   target：存放 Maven 构建当前模块，所生成的输出文件。
           classes：这就是 classpath。
                   com.**：存放编译后的 .class 文件
                   资源文件：src/main/resources 的所有文件。
   模块配置文件.iml
   ```

### 待解决

- 使用文件存储测试用例，txt文件读取时\n会消失好像
- 是否方便在不同的环境下

### 参考链接

- [Javadoc （Java API 文档生成器）详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/599276419)
- [Java文档注释用法+JavaDoc的使用详解-CSDN博客](https://blog.csdn.net/lsy0903/article/details/89893934)
- [小白都看得懂的Javadoc使用教程 - 说人话 - 博客园 (cnblogs.com)](https://www.cnblogs.com/linj7/p/14339381.html)
- [详解介绍JUnit单元测试框架（完整版）_junit整体框架详细介绍-CSDN博客](https://blog.csdn.net/qq_26295547/article/details/83145642)
- [IDEA中添加junit4的三种方法（详细步骤操作）_idea junit安装步骤-CSDN博客](https://blog.csdn.net/gakki_200/article/details/106413351)
- [看完这篇，别人的开源项目结构应该能看懂了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/115403195)
- [java项目文件夹结构_mob64ca12d12b68的技术博客_51CTO博客](https://blog.51cto.com/u_16213303/7548890#:~:text=文件夹结构 1 src ：存放项目的源代码。 2 lib ：存放项目所需的外部依赖库。 3,test ：存放项目的单元测试代码。 4 resources ：存放项目的配置文件和其他资源文件。 5 docs ：存放项目的文档。)
- [Maven是什么？有什么作用？Maven的核心内容简述_maven是干什么用-CSDN博客](https://blog.csdn.net/King_wq_2020/article/details/117818277#:~:text=Maven是一款服务于Java平台的自动化构建工具。 Maven,作为 Java 项目管理工具，它不仅可以用作包管理，还有许多的插件，可以支持整个项目的开发、打包、测试及部署等一系列行为。)
- [jar包的一些事儿 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/82320492)
- [Java的异常 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/1252599548343744/1264734349295520)
- [你一眼就看懂的IDEA中各种文件夹颜色标记的区别_idea左侧 test目录颜色-CSDN博客](https://blog.csdn.net/m0_46405589/article/details/108078898)
- [【Java工程目录结构】项目结构和模块结构_java项目文件结构-CSDN博客](https://blog.csdn.net/qq_48054091/article/details/134228891)
- [如何使用IDEA将代码打包成可执行jar - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/669542994)
- [IDEA 报错，无效的源发行版 无效的目标发行版 解决方法_无效的目标发行版: 20-CSDN博客](https://blog.csdn.net/qq_43362426/article/details/111370493)
- [IDEA生成可执行jar包及常见问题与解决_idea打包的jar包在哪里-CSDN博客](https://blog.csdn.net/qq_50918834/article/details/128067243)
- [Error: Registry key 'Software\JavaSoft\Java Runtime Environment'\CurrentVersion' has value '1.8', bu-CSDN博客](https://blog.csdn.net/superit401/article/details/70154993)

### 遇到问题

- 错误：找不到或无法加载主类

使用IDEA重新下载一个jdk就好了，也不知道为啥

- 警告: 使用不提供注释的默认构造器

好像跟类的重载有关系，还未解决

- JUnit提供了`Assert.assertEquals(double, double)`的重载方法`Assert.assertEquals(double, double, double)`
- 测试调用方法指定为静态，可以在不创建实例时直接使用
- Java的类方法中不支持默认参数，而是通过方法重载间接实现

### 笔记

##### java和C语言对比

- 丢弃了操作符重载、多继承、自动的强制类型转换。特别地，Java 语言不使用指针，而是引用。并提供了自动分配和回收内存空间
- java只支持类之间的单继承，但支持接口之间的多继承，而C语言只对虚函数动态绑定

##### 标识符

- 标识符：即类名、变量名以及方法名，以字母（A-Z或者a-z）,美元符($),下划线(_)开头

- 类名的每个单词首字母要大写，**MyFirstJavaClass** 

- 方法名都应该以小写字母开头，如**MyFirstJavaClass** 

- 源文件名与类名相同，注意大小写，因此在引用其他类时无需显式声明，编译器会根据类名去寻找同名文件

- 包名：所有字母小写，如```com.runoob```

- 常量：所有字母大写，每个单词之间用```_```链接

- 常见转义字符

  ```
  "\b" (退格)
  "\f" (换页)
  "\n" (换行)
  "\r" (回车)
  "\t" (水平制表符(到下一个tab位置))
  "\' " (单引号)
  "\" " (双引号) 
  "\\" (反斜杠)
  ```

##### 枚举

```
class FreshJuice {
   enum FreshJuiceSize{ SMALL, MEDIUM , LARGE }
   FreshJuiceSize size;
}
 
public class FreshJuiceTest {
   public static void main(String[] args){
      FreshJuice juice = new FreshJuice();
      juice.size = FreshJuice.FreshJuiceSize.MEDIUM  ;
   }
}
```

##### 数据类型

整数默认int类型，小数默认double类型

| 类型   | 默认值  | 字节 |
| ------ | ------- | ---- |
| bool   | false   | 1    |
| byte   | 0       | 1    |
| char   | ‘u0000’ | 2    |
| short  | 0       | 2    |
| int    | 0       | 4    |
| float  | 0.0f    | 4    |
| long   | 0L      | 8    |
| double | 0.0d    | 8    |

一个字节等于8位，等于256位数

一个英文字符或阿拉伯数字占1个字节

一个汉字占2个字节

- byte、int、long、和short都可以用十进制、16进制以及8进制的方式来表示。当使用字面量的时候，前缀 **0** 表示 8 进制，而前缀 **0x** 代表 16 进制, 例如：

```java
int decimal = 100;
int octal = 0144;
int hexa =  0x64;
```

- 自动类型转换

整型、常量、字符型数据能够混合运算，在运算中先转化为同一类型，从低到高转换：

```java
低  ------------------------------------>  高

byte,short,char—> int —> long—> float —> double 
```

而由高到低的转换需要使用强制类型转换，但有溢出和精度损失风险

```java
int i =128;   
byte b = (byte)i;
//注意：浮点数到整数的转换是舍弃小数部分
(int)23.7 == 23;        
(int)-45.89f == -45
```

- 其他类转化为string类

1. 调用类的串转换方法：X.toString();
2. 自动转换：X+"";
3. 使用String类的方法：String.valueOf(X);

- string类转化为其他类

1. 先转为封装类实例，再调用对应方法转换，如：new Float("32.1").doubleValue()或Double.valueOf("32.1").doubleValue()

2. 静态parse方法

   ```java
   String s = "1";
   byte b = Byte.parseByte( s );
   short t = Short.parseShort( s );
   int i = Integer.parseInt( s );
   long l = Long.parseLong( s );
   Float f = Float.parseFloat( s );
   Double d = Double.parseDouble( s );
   ```

##### 注释

1. 类注释

```java
/**
* Copyright (C), 2006-2010, ChengDu Lovo info. Co., Ltd.
* FileName: Test.java
* 类的详细说明
*
* @author 类创建者姓名
* @Date    创建日期
* @version 1.00
*/
```

2. 属性注释

```java
/** 提示信息 */
private String strMsg = null;
```

3. 方法注释

```java
/**
* 类方法的详细使用说明
*
* @param 参数1 参数1的使用说明
* @return 返回结果的说明
* @throws 异常类型.错误代码 注明从此类方法中抛出异常的说明
*/
```

4. 构造方法注释

```java
/**
* 构造方法的详细使用说明
*
* @param 参数1 参数1的使用说明
* @throws 异常类型.错误代码 注明从此类方法中抛出异常的说明
*/
```

5. 方法内部注释

单行或多行注释

```java
/**
* 构造方法的详细使用说明
*
* @param 参数1 参数1的使用说明
* @throws 异常类型.错误代码 注明从此类方法中抛出异常的说明
*/
```

##### java程序示例

一个源文件只能有**一个**public类和**多个**非public类，public类名与源文件名保持一致。代码顺序依次是：package - import - class

```java
package javawork.helloworld;
/*把编译生成的所有．class文件放到包javawork.helloworld中*/
import java awt.*;
//告诉编译器本程序中用到系统的AWT包
import javawork.newcentury;
/*告诉编译器本程序中用到用户自定义的包javawork.newcentury*/
 public class HelloWorldApp{...｝
/*公共类HelloWorldApp的定义，名字与文件名相同*/ 
class TheFirstClass｛...｝;
//第一个普通类TheFirstClass的定义 
interface TheFirstInterface{......}
/*定义一个接口TheFirstInterface*/
```

##### 变量

- **局部变量**：在方法、构造方法或者语句块中定义的变量被称为局部变量。变量声明和初始化都是在方法中，方法结束后，变量就会自动销毁。
- **成员变量**：成员变量是定义在类中，方法体之外的变量。这种变量在创建对象的时候实例化。成员变量可以被类中方法、构造方法和特定类的语句块访问。
- **类变量**：类变量也声明在类中，方法体之外，但必须声明为 static 类型，实质上是一个全局变量，所有对象共享，同时可以被类名调用。

##### main为什么在类中？

Java要求所有程序都放在类对象中，所以main函数寄居在某个class中，作为Java程序的总入口

##### 异常处理

- [Java 中 try-catch,throw和throws的使用_try catch-CSDN博客](https://blog.csdn.net/Sundy_sc/article/details/101106847)

##### Junit

一般的单元测试

```java
@Test
    public void testAdjustTaxRates() {
        String[] inputs = {"ab\n3\ncd\n0.15","-10\n4\n-0.74\n0.6\n","6\n0.37\n","8\n7\n0.55\n"}; // 模拟多个用户输入
        double[] expectedRate = {0.15,0.6,0.37,0.55}; // 期望的税额
        int [] checkRate = {3,4,6,7}; // 更新的级数
        for (int i = 0; i < inputs.length; i++) {
            System.setIn(new ByteArrayInputStream(inputs[i].getBytes()));
            Scanner scanner = new Scanner(System.in);
            PersonalTaxCalc.adjustTaxRates(scanner);
            assertEquals(expectedRate[i], PersonalTaxCalc.taxRates[checkRate[i]-1],1e-6);
        }
    }
```

使用反射机制测试私有方法和私有变量

```java
@Test
    public void testgetInputValue() throws Exception{
        Method getInputValue = PersonalTaxCalc.class.getDeclaredMethod("getInputValue",
                Scanner.class, String.class);
        Method getInputValueRate = PersonalTaxCalc.class.getDeclaredMethod("getInputValue",
                Scanner.class, String.class, boolean.class);
        getInputValue.setAccessible(true); // 设置为可访问
        getInputValueRate.setAccessible(true);

        String[] inputs = {"-10\n5000\n","5000\n","ab\n5000\n"}; // 模拟多个用户输入
        double[] expectedValue = {5000.0,5000.0,5000.0}; // 期望的税额
        for (int i = 0; i < inputs.length; i++) {
            System.setIn(new ByteArrayInputStream(inputs[i].getBytes()));
            Scanner scanner = new Scanner(System.in);
            double actualValue =  (double)getInputValue.invoke(null,scanner,"输入文本:"); 
//            double actualValue = PersonalTaxCalc.getInputValue(scanner, "输入文本:");
            assertEquals(expectedValue[i], actualValue,1e-6); // 断言实际输入值与期望值相等
        }
        String[] inputsRate = {"-10\n3\n","3\n","11\n6\n"}; // 模拟多个用户输入
        double[] expectedValueRate = {3.0,3.0,6.0}; // 期望的税额
        for (int i = 0; i < inputsRate.length; i++) {
            System.setIn(new ByteArrayInputStream(inputsRate[i].getBytes()));
            Scanner scanner = new Scanner(System.in);
            double actualValue =  (double)getInputValueRate.invoke(null,scanner,"输入文本:",true);
//            double actualValue = PersonalTaxCalc.getInputValue(scanner, "输入文本:", true);
            assertEquals(expectedValueRate[i], actualValue,1e-6); // 断言实际输入值与期望值相等
        }
    }
```

读取测试用例文件进行测试

```java
@Test
    public void testAdjustThreshold() throws IOException {
        Path filePath = Paths.get("test/resources/testcases_adjustThreshold.txt"); 
        List<String> testCases = Files.readAllLines(filePath, Charset.defaultCharset());

        for (String testCase : testCases) {
            String[] testData = testCase.split(","); // 测试数据以逗号分隔
            String inputValue = testData[0] + "\n";
            for (int i = 0; i < testData.length-1; i++ ){
                inputValue = inputValue + testData[i] + "\n";
            }
            double expectedThreshold = Double.parseDouble(testData[testData.length-1]);
            System.setIn(new ByteArrayInputStream((inputValue).getBytes()));
            Scanner scanner = new Scanner(System.in);

            PersonalTaxCalc.adjustThreshold(scanner);
            assertEquals(expectedThreshold, PersonalTaxCalc.threshold, 1e-6); 
        }
```



