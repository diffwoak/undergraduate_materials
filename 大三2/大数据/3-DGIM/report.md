## Assignment3：实现DGIM算法

### 作业要求

1. 实现DGIM算法，在给定的长度为100万的二值数值流（存储在TestDataStream.txt），完成算法测试

2. 设置滑动窗口大小为1万，运行DGIM算法

3. 通过实验分析每一个滑动窗口的实验误差，即对比该窗口的1的实际数量（存储在Groundtruth.txt）和DGIM输出的1的数量，计算所有窗口的实验误差平均值和和方差

### 作业过程

#### 一、 实现DGIM算法

主要功能在于：

1. 读入字符 1时添加一个bucket
2. 每次读完字符后都要检测最早的bucket是否需要丢弃
3. 检测是否存在相同大小的bucket，有则合并

定义一个DGIM类，其中主要代码为：

```python
def update_buckets(self, timestamp, bit):
    # 读入字符 1时添加一个bucket
    if int(bit) == 1:self.buckets.append((timestamp, 1))
	# 每次读完字符后都要检测最早的bucket是否需要丢弃
    while len(self.buckets) > 0 and timestamp - self.buckets[0][0] >= self.window_size:
        self.buckets.pop(0)
    # 检测是否存在相同大小的bucket，有则合并
    i = 0
    while i < len(self.buckets) - 1:
        if self.buckets[i][1] == self.buckets[i + 1][1]:
            self.buckets[i] = (self.buckets[i][0], self.buckets[i][1] + self.buckets[i + 1][1])
            self.buckets.pop(i + 1)
        else:
            i += 1
```

主函数：

```python
def main():
    window_size = 10000
    dgim = DGIM(window_size)
    timestamp = 1
    errors = []
    with open('TestDataStream.txt', 'r') as data_file, open('Groundtruth.txt', 'r') as groundtruth_file:
        bit = data_file.readline().strip()		# 逐行读取0/1
        # 循环读入数据流直到数据流充满窗口
        while bit and timestamp < window_size:	
            dgim.update_buckets(timestamp,bit)
            timestamp += 1
            bit = data_file.readline().strip()
        truth = groundtruth_file.readline().strip()
        # 循环读入数据流和窗口的实际1数量，进行比对
        while bit:								
            dgim.update_buckets(timestamp,bit)
            ones_count = dgim.count_ones()		# 计算DGIM的1数量
            dgim_error = abs(ones_count - int(truth))	# 计算误差
            timestamp += 1
            bit = data_file.readline().strip()
            truth = groundtruth_file.readline().strip()
            errors.append(dgim_error)
            if timestamp >= window_size and timestamp % 50000 == 0:
                print(f"Timestamp: {timestamp}, DGIM Count: {ones_count}, Groundtruth: {truth}, Error: {dgim_error}")
        # 计算均值、方差
        mean = np.mean(errors)
        variance = np.var(errors)
        print(f"mean: {mean}, var: {variance}")
```

#### 二、 运行测试

<img src="https://gitee.com/e-year/images/raw/master/img/202404091635788.png" alt="image-20240329200928343" style="zoom:50%;" />

所有窗口的实验误差平均值为2048.5273，方差为1396347.67。

### 总结

本次作业通过代码简单实现了DGIM算法，对数据流统计的体会加深，还有待改进的地方，比如存储每个bucket的size使用2的次数直接表示，优化buckets的存储空间；调整窗口大小分析对精度的影响。