### 大数据原理与技术 平时作业3数据格式说明

#### 1. TestDataStream.txt

共$N=1,000,000$行，每一行记录一个二进制流数据。



#### 2. Groundtruth.txt

设滑动窗口大小为$W=10,000$，记录每一个滑动窗口内1的个数。

共有$N-W+1$行，第$i$行记录了第$i$个滑动窗口$\left[i,i+W-1\right]$中1的数量。



请参考`TestDataStream_example.txt`和`Groundtruth_example.txt`，这是一个$N=20$，$W=5$时的样例。

