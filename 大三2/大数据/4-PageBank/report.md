## Assignment4：实现PageRank算法的Google快速版

### 作业要求

1. 实现PageRank算法的Google快速版，在给定的具有7115个节点、103689条边的有向图（存储在dataset.txt），完成算法测试
2. 计算得到每一个node的PageRank score

### 作业过程

#### 一、 实现PageRank算法

主要功能在于：

1. 读文件，初始化encode sparse matrix: $$M$$,迭代PageRank使用的$$r^{new},r^{old}$$
2. 计算$$r^{new}=(\beta M+(1-\beta)/N)r^{old}$$迭代$$r^{new}$$和$$r^{old}$$，直至$$|r^{new}-r^{old}|<e$$

代码实现：

1. 初始化变量

​	将文件中所有出现的节点初始化于$$M、r^{new}、r^{old}$$中

```python
M = {}    
R_new, R_old = {},{}
with open('dataset.txt', 'r') as data_file:
        for line in data_file:
            if line[0] == '#': continue
            key, value = line.split()
            if key not in M:
                M[key] = [value]
                R_old[key] = 1/num
                R_new[key] = (1-beta)/num
            else:
                M[key].append(value)
            if value not in M:
                M[value] = []
                R_old[value] = 1/num
                R_new[value] = (1-beta)/num
```

2. 循环迭代$$r^{new}、r^{old}$$至收敛

```python
	epoch = 0
    while epoch < 500:
        # initialize R_new
        for i in R_new:
            R_new[i] = (1-beta)/num
        # calculate R_new
        for i in R_old:
            d_i = len(M[i])
            for dest in M[i]:
                R_new[dest] += beta*(R_old[i]/d_i)
        # calculate difference
        diff_sum = 0
        for i in R_old:
            diff_sum += np.abs(R_old[i] - R_new[i])
            R_old[i] = R_new[i]
        print(f"the diff in epoch {epoch} is {diff_sum}")
        # determine whether to break the loop
        if diff_sum < e: break
        epoch += 1
```

3. 输出结果

​	将结果写入文件保存

```python
with open('result.txt', 'w') as file:
        file.write(f"page\t rank\n")
        for key, values in R_new.items():
            file.write(f"{key}\t {values}\n")
```

#### 二、 运行测试

设置参数

```python
num = 7115	# num of nodes
beta = 0.8	# probability of follow a link
e = 10e-50	# error bound
```

运行结果

<img src="https://gitee.com/e-year/images/raw/master/img/202404121335940.png" alt="image-20240412133520958" style="zoom:67%;" />

<img src="https://gitee.com/e-year/images/raw/master/img/202404121336880.png" alt="image-20240412133603360" style="zoom: 67%;" />

### 总结

本次作业简单实现了PageRank算法，在课上对PageRank伪代码了解后总体写下来比较顺畅，主要熟悉了算法的代码实现，实现过程通过构造课件上的例子比对PageRank的结果来确保代码的正确性。

