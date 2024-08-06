## Lab3 - Pthreads并行矩阵乘法与数组求和

### 实验要求

**一、并行矩阵乘法**

使用Pthreads实现并行矩阵乘法，并通过实验分析其性能。

输入：m, n, k 三个整数，每个整数取值范围均为[128, 2048]

问题描述：随机生成m×n的A矩阵及n×k的B矩阵，矩阵相乘得到C矩阵

输出：A, B, C三个矩阵，以及矩阵计算时间 t

要求：

1. 使用Pthread创建多线程实现并行矩阵乘法，调整线程数量（1-16）及矩阵规模（128-2048），根据结果分析其并行性能（包括但不限于，时间、效率、可扩展性）。

2. 选做：可分析不同数据及任务划分方式的影响。

**二、并行数组求和**

使用Pthreads实现并行数组求和，并通过实验分析其性能。

输入：整数n，取值范围为[1M, 128M]

问题描述：随机生成长度为n的整型数组$A$，计算其元素和 $s=\sum_{i=1}^{n}A_i$

输出：数组A，元素和s，及求和计算所消耗的时间 t

要求：

1. 使用Pthreads 实现并行数组求和，调整线程数量（1-16）及数组规模（1M，128M），根据结果复习其并行性能（包括但不限于，时间、效率、可扩展性）。
2. 选做：可分析不同聚合方式的影响。

### 实验过程

#### 一、并行矩阵乘法

##### 1. 实现思路

与上一个MPI实验内容相似，将A划分到各个线程与矩阵B进行矩阵相乘，再汇总到矩阵C中

##### 2. 代码实现

定义结构体存储每个线程的处理数据

```c
struct ThreadData {
    double* A;
    double* B;
    double* C;
    int m;
    int n;
    int k;
    int start;
    int end;
};
```

计算分配到每一个线程的数据范围，创建线程

```c
void matrix_multiple(double* A, double* B, double* C, int m, int n, int k, int num_threads) {
    pthread_t threads[num_threads];
    struct ThreadData thread_data[num_threads];

    int rows_per_process = m / num_threads;
    int remaining_rows = m % num_threads;

    int offset = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].m = m;
        thread_data[i].n = n;
        thread_data[i].k = k;
        thread_data[i].start = offset;
        offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
        thread_data[i].end = offset;
        pthread_create(&threads[i], NULL, matrix_multiple_thread, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
```

线程执行

```c
void* matrix_multiple_thread(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    printf("I'm thread begin with %d \n", data->start);
    for (int B_i = 0; B_i < data->n; B_i++) {
        for (int A_i = data->start; A_i < data->end; A_i++) {
            data->C[A_i * data->n + B_i] = 0;
            for (int A_j = 0; A_j < data->k; A_j++) {
                data->C[A_i * data->n + B_i] += data->A[A_i * data->k + A_j] * data->B[A_j * data->n + B_i];
            }
        }
    }
    pthread_exit(NULL);
}
```

##### 3. 运行结果

```shell
gcc -g -Wall -o pth pmatrix.c -lpthread
./pth 4
```

<img src="https://gitee.com/e-year/images/raw/master/img/202404090019479.png" alt="image-20240408232822515" style="zoom:67%;" />

##### 4. 性能分析

记录不同线程数量（1-16）及矩阵规模（128-2048）下的时间开销，耗时单位毫秒。

| 线程数\矩阵规模 |  128   |   256   |   512    |   1024    |    2048    |
| :-------------: | :----: | :-----: | :------: | :-------: | :--------: |
|        1        | 21.616 | 275.948 | 2138.426 | 27905.705 | 267350.834 |
|        2        | 21.160 | 270.047 | 2486.53  | 29149.927 | 275.937894 |
|        4        | 34.696 | 308.223 | 2384.006 | 28602.850 | 322226.751 |
|        8        | 25.558 | 301.801 | 2803.819 | 30186.458 | 341741.185 |
|       16        | 34.326 | 313.104 | 2804.56  | 33789.139 | 354035.642 |

可以看到，随着线程数量增加，同一规模下的耗时并未减少，可见线程调度的开销要高于多线程计算的优势。

#### 二、并行数组求和

##### 1. 实现思路

将数组A按线程数平均划分，每个线程对其范围内的元素进行加和，再进行最终汇总操作。

##### 2. 代码实现

定义结构体存储每个线程的处理数据

```c
struct ThreadData {
    int* A;
    int start;
    int end;
    long long sum;
};
```

每个线程计算该范围内加和

```
void* partial_sum(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    data->sum = 0;
    for (int i = data->start; i < data->end; i++) {
        data->sum += data->A[i];
    }
    pthread_exit(NULL);
}
```

3. ##### 运行结果

```
gcc -g -Wall -o ps psum.c -lpthread
./ps 4
```

![image-20240408220509640](https://gitee.com/e-year/images/raw/master/img/202404090019411.png)

4. ##### 性能分析

| 线程数\矩阵规模 |   1M   |   4M    |   16M   |   64M    |   128M   |
| :-------------: | :----: | :-----: | :-----: | :------: | :------: |
|        1        | 7.909  | 22.321  | 68.202  | 247.045  | 504.990  |
|        2        | 23.888 | 74.621  | 161.913 | 662.871  | 1936.196 |
|        4        | 32.431 | 69.713  | 486.624 | 1491.387 | 2534.987 |
|        8        | 38.310 | 113.104 | 380.524 | 1226.916 | 3272.917 |
|       16        | 38.871 | 116.222 | 313.567 | 886.434  | 1625.301 |

观察数据，随着并行线程数上升，数组求和的耗时没有下降，可见线程调度开销也大于并行计算优势，但在线程数上升到16时，在较大规模的计算（如16M以上）中耗时少于线程数为8的计算，在同样具有很大线程调度开销的情况下，16线程数带来的并行计算优势才慢慢显现。

##### 5. 改进

实验完后又做了一些改进，使用互斥锁方式在每个线程运行时修改sum，避免等待所有线程结束后才执行sum的汇总，节省时间

```c
void* partial_sum(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    long long tmp = 0;
    for (int i = data->start; i < data->end; i++) {
        tmp += data->A[i];
    }
    pthread_mutex_lock(&mutex);
    sum += tmp;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}
```

修改后统计数据如下，可看出在效率方面有很大提升

| 线程数\矩阵规模 |   1M   |   4M   |  16M   |   64M   |  128M   |
| :-------------: | :----: | :----: | :----: | :-----: | :-----: |
|       16        | 19.732 | 49.488 | 74.409 | 291.795 | 996.796 |

