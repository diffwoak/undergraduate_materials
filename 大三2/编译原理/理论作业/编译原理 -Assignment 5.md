### 编译原理 -Assignment 5

---

一、（18 分）给按要求给出如下基本块（Basic Block）的局部优化（Local Optimization）：

1、（6 分）为以下基本块(basic block) 消除公共子表达式，并在原指令的右边空白处写上优化后调整

的指令(不必重写那些不变的指令)。做此题时请勿应用其他的优化技术。

![image-20240617112807178](C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\image-20240617112807178.png)

g = a + n

​		$$t1 = b * c$$

​		$$u = g$$

t = a * a

a = 2 + a

​		$$h = t1$$

2、（6 分）针对以下代码片段反复施用复制传播 (copy propagation，如a = b; c = a;可优化为c=b;)、常量折叠 (constant folding)以及代数化简(algebraic simplification，即利用代数恒等式)，并在原指令的右边空白处写上优化后调整的指令 (不必重写那些不变的指令)。做此题时请勿应用其他的优化技术。

![image-20240617112752502](C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\image-20240617112752502.png)

c = 3

o = a

​		$$m = b + a$$

​		$$p = 3*3$$ $$\rightarrow p = 9$$

​		$$i = x * 0$$  $$\rightarrow i = 0$$

n = m * 2

​		$$e = 3 + 9$$   $$ \rightarrow e = 12$$

​		$$r = 12 + n$$

3、（6 分）在以下基本块中的所有程序点 (program point，见括号) 处填写活跃变量 (live variable) 集。提示:在基本块入口与出口处的活跃变量集分别是{x}与{s,e}，且基本块中不包含死代码。

![image-20240617112729572](C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\image-20240617112729572.png)
$$
\{x\} \\
b = x * 0 \\
	\{b \}\\
n = b \\
	\{n,b\}\\
a = b * 2 \\
	\{a,n\} \\
e = a + n \\
	\{e\} \\
s = 2 + 3 \\
	\{s,e\}
$$

---

二、（34 分）给定如下中间代码的基本块（Basic Block）：

![image-20240617112711385](C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\image-20240617112711385.png)

1、（18 分）构造该基本块的有向无环图（Directed Acyclic Graph，简称 DAG）。

<img src="C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\9b299c3fb11e0dcbab1b02faa0beaf4.jpg" alt="9b299c3fb11e0dcbab1b02faa0beaf4" style="zoom:33%;" />

2、（16 分）分别有如下假设：

1）假设#1：仅变量 a 在基本块的出口（exit）是活跃的（live）；

2）假设#2：变量 f 和 a 在基本块的出口均是活跃的。

试分上述 2 种不同的假设情况，分别基于你构造出来的 DAG 对基本块进行优化

1) 假设#1
   - f不是活跃变量，可删去
   - d,b有公共子表达式，可将d删去

2. 假设#2
   - d,b有公共子表达式，将d删去

---

三、（48 分）给定如下中间代码片段：

![image-20240617112652219](C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\image-20240617112652219.png)

1、（18分）将上述代码片段划分基本块（Basic Block），并画出该代码片段的流图（Flow Graph）。你可以直接画出流图，在图中的每一结点中用n–m表示该基本块由第n至m条指令组成。

<img src="C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\fe19dbdd7a083c23840bcbdcd16950a.jpg" alt="fe19dbdd7a083c23840bcbdcd16950a" style="zoom:33%;" />

2、（20分）为实现代码片段7–11的局部优化（Local Optimization），请将此段代码转换为一个有向无环图（Directed Acyclic Graph，简称DAG）。

<img src="C:\Users\asus\Desktop\大三下\编译原理\理论作业\编译原理 -Assignment 5.assets\500f2bcb3da8c2d3cf759908a889c0b.jpg" alt="500f2bcb3da8c2d3cf759908a889c0b" style="zoom:33%;" />

3、（10分）对代码片段7–11，指出其中的两种代码优化方法。

- 常量折叠和传播：直接用$$c=2$$代替$$c = 4/2$$，用$$t2 = 1$$代替$$t2 = c -1$$
- 消除死代码：$$t1$$不是活跃变量，可删除$$t1 = x * c$$
