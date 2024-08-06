## 编译原理 -Assignment 1

1. **RE = 1\*（0|111\*）\*1*** represents all strings that do not contain 010

   1. Using **Thompson Algorithm** to construct a finite automaton

      <img src="https://gitee.com/e-year/images/raw/master/img/202403091722033.png" alt="graphviz" style="zoom: 200%;" />

   2. Converting NFA to DFA if the FA obtained in 1.1 is nondeterministic

      | $$I$$                                  | $$I_1$$                                                      | $$I_0$$                              | Accept |
      | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------ | ------ |
      | {1,2,4,5,6,7,15,18,19,20,22}mark$$T0$$ | {3,8,21,2,4,5,6,7,9,15,18,19,20,21,22}mark $$T1$$            | {16,6,7,15,17,18,19,20,22}mark$$T2$$ | Yes    |
      | $$T1$$                                 | {3,8,10,21,2,4,5,6,7,9,11,12,14,15,17,18,19,20,21,22}mark$$T3$$ | $$T2$$                               | Yes    |
      | $$T2$$                                 | {8,21,9,20,22}mark$$T4$$                                     | $$T2$$                               | Yes    |
      | $$T3$$                                 | {3,8,10,13,21,2,4,5,6,7,9,11,12,14,15,17,18,19,20,21,22}mark$$T5$$ | $$T2$$                               | Yes    |
      | $$T4$$                                 | {10,21,6,7,11,12,14,15,17,18,19,20,22}mark$$T6$$             |                                      | Yes    |
      | $$T5$$                                 | $$T5$$                                                       | $$T2$$                               | Yes    |
      | $$T6$$                                 | {8,13,21,6,7,9,14,15,17,18,19,20,22}mark$$T7$$               | $$T2$$                               | Yes    |
      | $$T7$$                                 | {8,10,21,6,7,9,11,12,14,15,17,18,19,20,22}mark $$T8$$        | $$T2$$                               | Yes    |
      | $$T8$$                                 | {8,10,13,21,6,7,9,11,12,14,15,17,18,19,20,22}mark$$T9$$      | $$T2$$                               | Yes    |
      | $$T9$$                                 | $$T9$$                                                       | $$T2$$                               | Yes    |

      ![graphviz (2)](https://gitee.com/e-year/images/raw/master/img/202403091722875.png)

   3. Determine whether the DFA obtained in 1.2 is minimized. If not, please minimize the DFA

      intial set:{0,1,2,3,4,5,6,7,8,9}

      8 on '1' $$\rightarrow$$ 9,8 on '0' $$\rightarrow$$ 2;9 on '1' $$\rightarrow$$ 9;9 on '0' $$\rightarrow$$ 2 $$\Rightarrow$${0,1,2,3,4,5,6,7,89}

      7 on '1' $$\rightarrow$$ 89,7 on '0' $$\rightarrow$$ 2;89 on '1' $$\rightarrow$$ 89;89 on '0' $$\rightarrow$$ 2 $$\Rightarrow$${0,1,2,3,4,5,6,789}

      同理可得到{0,1,2,3,4,6789}

      3 on '1' $$\rightarrow$$ 5,3 on '0' $$\rightarrow$$ 2;5 on '1' $$\rightarrow$$ 5;5 on '0' $$\rightarrow$$ 2 $$\Rightarrow$${0,1,2,35,4,6789}

      同理可得到{0135,2,4,6789}

      0135 on '1' $$\rightarrow$$ 0135,0135 on '0' $$\rightarrow$$ 2;6789 on '1' $$\rightarrow$$ 6789;6789 on '0' $$\rightarrow$$ 2 $$\Rightarrow$${01356789,2,4}

   ![graphviz (3)](https://gitee.com/e-year/images/raw/master/img/202403091722002.png)

2. The number of character '0' in each string is a multiple of three(including zero)

   1. Provide the RE with reason

      RE = (1\*01\*01\*01\*)*

      对于1\*01\*01\*01\*，是包含3个0的任意字符串，再其使用kleen closure，则能得到任意具有3的倍数的字符串

   2. Construct the corresponding DFA

      先转为NFA

      ![graphviz (4)](https://gitee.com/e-year/images/raw/master/img/202403091722202.png)

      转为DFA

      | $$I$$                  | $$I_1$$                         | $$I_0$$                          | Accept |
      | ---------------------- | ------------------------------- | -------------------------------- | ------ |
      | {0,1,2,4,20}mark$$T0$$ | {3,2,4}mark $$T1$$              | {5,6,7,9}mark$$T2$$              | Yes    |
      | $$T1$$                 | $$T1$$                          | $$T2$$                           | No     |
      | $$T2$$                 | {8,7,9}mark$$T3$$               | {10,11,12,14}mark$$T4$$          | No     |
      | $$T3$$                 | $$T3$$                          | $$T4$$                           | No     |
      | $$T4$$                 | {12,13,14}mark$$T5$$            | {15,16,17,19,20,1,2,4}mark$$T6$$ | No     |
      | $$T5$$                 | $$T5$$                          | $$T6$$                           | No     |
      | $$T6$$                 | {3,18,17,19,20,1,2,4}mark$$T7$$ | $$T2$$                           | Yes    |
      | $$T7$$                 | $$T7$$                          | $$T2$$                           | Yes    |

      ![graphviz (8)](https://gitee.com/e-year/images/raw/master/img/202403091722475.png)

      最小化DFA

      intial set:{0,6,7} {1,2,3,4,5}

      for {0,6,7}

      6 on '1' $$\rightarrow$$ 7,6 on '0' $$\rightarrow$$ 2;7 on '1' $$\rightarrow$$ 7;7 on '0' $$\rightarrow$$ 2 $$\Rightarrow$${0,67}

      for {1,2,3,4,5}

      4 on '1' $$\rightarrow$$ 5,4 on '0' $$\rightarrow$$ 6;5 on '1' $$\rightarrow$$ 5;5 on '0' $$\rightarrow$$ 6 $$\Rightarrow$${1,2,3,45}

      2 on '1' $$\rightarrow$$ 3,2 on '0' $$\rightarrow$$ 4;3 on '1' $$\rightarrow$$ 3;3 on '0' $$\rightarrow$$ 4 $$\Rightarrow$${1,23,45}

      final:{0,67},{1,23,45}

      ![graphviz (7)](https://gitee.com/e-year/images/raw/master/img/202403091722731.png)

### 参考链接

- [绘制的graphviz图怎么在hexo和typora上都显示 · Issue #28 · hyeoy/mynote · GitHub](https://github.com/hyeoy/mynote/issues/28)

### 画图代码

- [Graphviz Online (dreampuf.github.io)](https://dreampuf.github.io/GraphvizOnline/#digraph G {   subgraph cluster_0 {    style%3Dfilled%3B    color%3Dlightgrey%3B    node [style%3Dfilled%2Ccolor%3Dwhite]%3B    a0 -> a1 -> a2 -> a3%3B    label %3D "process %231"%3B  }   subgraph cluster_1 {    node [style%3Dfilled]%3B    b0 -> b1 -> b2 -> b3%3B    label %3D "process %232"%3B    color%3Dblue  }  start -> a0%3B  start -> b0%3B  a1 -> b3%3B  b2 -> a3%3B  a3 -> a0%3B  a3 -> end%3B  b3 -> end%3B   start [shape%3DMdiamond]%3B  end [shape%3DMsquare]%3B })
- [正则表达式转自动机图形化显示的在线工具 (misishijie.com)](https://misishijie.com/tool/index.html)
- [labeltarget | Graphviz](https://graphviz.gitlab.io/docs/attrs/labeltarget/)

```
digraph G {
    rankdir = LR
    node [shape = plaintext]
    start;
    node [width=0.5,height=0.5]
    node [shape = doublecircle]
    22;
    node [shape = circle]
    start -> 1;
    1 -> 2 [label = ε]
    2 -> 3 [label = 1]
    3 -> 2 [label = ε]
    3 -> 4 [label = ε]
    5 -> 6 -> 7 [label = ε]
    6 -> 7 [label = ε]
    7 -> 8 [label = 1]
    8 -> 9 [label = ε]
    9 -> 10 [label = 1]
    10 -> 11 -> 12 [label =ε]
    12 -> 13 [label = 1]
    13 -> 12 [label = ε]
    13 -> 14 -> 17 -> 6 [label = ε]
    17 -> 18 [label = ε]
    6 -> 15 [label = ε]
    15 -> 16 [label = 0]
    16 -> 17 [label = ε]
    19 -> 20 [label = ε]
    20 -> 21 [label = 1]
    21 -> 20 [label = ε]
    21 -> 22 [label = ε]
     1 -> 4 -> 5 -> 18 -> 19 -> 22 [label = ε]
    }
```

