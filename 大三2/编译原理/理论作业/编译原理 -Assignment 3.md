### 编译原理 -Assignment 3

---

**Q1:**在设计递归下降预测翻译器(Recursive Descent Predictive Translator)时，会让每一个非终结符号 A对应一个递归函数，其中A的每一个继承属性都对应着该函数的一个（ **参数** ），该函数的返回值则是A的（ **综合属性** ）。

---

**Q2:** 如果一个翻译模式(Translation Scheme)中存在嵌入在产生式右部的左边或中间的语义动作, 该翻译模式可通过哪些3种翻译技术实现? 

LL RDP递归下降预测翻译、LL非递归预测翻译、LR分析技术

---

**Q3:** 如下翻译模式(Translation Scheme): 

$$S\rightarrow \{A_1.i:=5\}A_1\{A_2.i:=6\}A_2$$

$$A\rightarrow a \{print(A.i)\}$$

对于输入串 a a, 执行该翻译模式的打印结果是什么? (5分)

打印结果：5 6

---

给定如下语言定义:

$$S\rightarrow (L)|a$$

$$L\rightarrow L,S|S$$

**Q4:** 给出一个语法制导定义(SDD), 计算输入串中配对括号的个数, 结果作为文法开始符号S的一个综合属性值, 并利用print()函数打印。(15分)

• 例如，输入的句子为 ( ( a , ( a ) ) , ( a , ( a , a , a ) ) , (a) ), 则输出结果为 6

$$S^{'} \rightarrow S\ \{print(S.count)\}$$

$$S\rightarrow (L)\ \{S.count = L.count+1\}$$

$$S\rightarrow a\ \{S.count = 0\}$$

$$L\rightarrow L_1,S\ \{L.count = L_1.count+S.count\}$$

$$L \rightarrow S\ \{L.count = S.count\}$$

**Q5:** 设计一个SDT, 打印出输入串中每一个a的嵌套深度。(20分)

• 例如， 输入的句子为 (a, ( a , ( a ) ) , ( a , ( a ) ) , a ), 则输出结果为 1 2 3 2 3 1

$$S\rightarrow (\{L.depth = S.depth + 1\}L)$$

$$S\rightarrow a\ \{print(S.depth)\}$$

$$L\rightarrow \{L_1.depth = L.depth\}L_1,\{S.depth = L.depth\}S $$

$$L\rightarrow \{S.depth = L.depth\}S$$

---

如下语言定义一个表达式的开销 (cp) = 其所有子表达式的开销 + 其本身的开销

- 加法的开销是1，乘法的开销是2，顺序操作的开销是0，如 (a + E).cp = 1 + a.cp + E.cp

- 赋值的开销是1， 还包括作为右值的子表达式的开销

- 循环的开销是3 ，还包括循环中每一次迭代的子表达式开销，即子表达式.cp * 循环次数 + 3

- 每一个变量 id 与常量 int 作为表达式的开销均为1,例如: 产生式Expr → int与Expr → id的语义规则均为Expr.cp := 1; 

$$
Expr\rightarrow for\ id:=int_1\ to\ int_2\ do\ Expr_1
	\\ |\ id:=\ Expr_1(赋值)
	\\ |\ Expr_1;Expr2(顺序操作)
	\\ |\ Expr_1\ *\ Expr2(乘法)
	\\ |\ Expr_1\ +\ Expr_2(加法)
	\\ |\ id (变量表达式)
	\\ |\ int (常量表达式)
$$

**Q6:** 给出一个计算上述开销函数定义的语法制导定义(SDD)。可假设有一个属性 val 包含单词的词法分析值; 你也可按自己的需要定义其他属性。(40分)

$$Expr\rightarrow for\ id:=int_1\ to\ int_2\ do\ Expr_1 \{Expr.cp = Expr_1.cp*(int_2.val-int_1.val)+3\}$$

$$Expr\rightarrow id:=\ Expr_1\{Expr.cp = Expr_1.cp+1\}$$

$$Expr\rightarrow Expr_1;Expr2\{Expr.cp = Expr_1.cp+Expr_2.cp\}$$

$$Expr\rightarrow Expr_1\ *\ Expr2\{Expr.cp = Expr_1.cp+Expr_2.cp+2,\}$$

$$Expr\rightarrow Expr_1\ +\ Expr_2\{Expr.cp = Expr_1.cp+Expr_2.cp+1\}$$

$$Expr\rightarrow id\{Expr.cp = 1\} $$

$$Expr\rightarrow int\{Expr.cp = 1\}$$

**Q7:** 请说明你在语法制导定义中引入的每一个属性是综合属性还是继承属性。(10分)

上述SDD中的$$Expr.cp$$和$$int.val$$都是综合属性
