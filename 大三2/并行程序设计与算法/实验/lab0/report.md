## Lab0 - 环境设置与串行矩阵乘法

### 实验要求

实现串行矩阵乘法，并通过对比实验分析性能

**输入：**m, n, k三个整数，取值范围均为[ 512, 2048]

**问题描述：**随机生成m x k 的A矩阵和k x n的B矩阵，矩阵乘法后得到C矩阵

**输出：**A，B，C三个矩阵，以及矩阵运算所耗时间

**要求：**实现多个版本的对比，涉及python、C、Intel MKL三个版本的代码

### 实现过程

##### 1. C语言实现

矩阵乘法过程为：最外层循环取出B的每一列，存储在tmp中（根据空间局部性，将列存储转为行存储，避免重复取一个列中的数），然后循环取出A的每一行，与tmp向量相乘存入C中对应位置，主要代码如下。

```C
void matrix_multiple(double* A,double* B,double* C,int m,int n,int k){
    for(int B_i = 0;B_i< n;B_i++){
        double* tmp = (double *)malloc( k*sizeof( double ));
        for(int B_j = 0;B_j < k;B_j++){
            tmp[B_j] = B[B_j*n+B_i];
        }
        for(int A_i = 0; A_i < m;A_i++){
            for(int A_j = 0; A_j < k;A_j++){
                C[A_i*n+B_i]+=A[A_i*k+A_j]*tmp[A_j];
            }
        }
        free(tmp);
    }
}
```

编译运行

```shell
gcc matrix.c -o matrix.out
./matrix.out
```

<img src="https://gitee.com/e-year/images/raw/master/img/202403191933275.png" alt="image-20240319193329729" style="zoom: 67%;" />

##### 2. python实现

与C语言的实现思路相同，便于进行比较

```python
def matrix_multiple(A,B,C,m,n,k):
    for B_i in range(n):
        tmp = []
        for B_j in range(k):
            tmp.append(B[B_j*n+B_i])
        for A_i in range(m):
            for A_j in range(k):
                C[A_i*n+B_i]+=A[A_i*k+A_j]*tmp[A_j]
    return C
```

运行

```shell
# 安装numpy库
sudo apt-get install python3-numpy 
# 运行
python3 matrix.py
```

<img src="https://gitee.com/e-year/images/raw/master/img/202403191947241.png" alt="image-20240319194711435" style="zoom:67%;" />

##### 3. 调整循环顺序

实际上就是将比对行优先遍历与列优先遍历，在上面C语言的实现中已经考虑到这个因素考虑行优先遍历，在此修改为列优先遍历，功能改为如下，不存储B的列到tmp，重复查询。

```C
void matrix_multiple_colum(double* A,double* B,double* C,int m,int n,int k){
    for(int B_i = 0;B_i< n;B_i++){
        for(int A_i = 0; A_i < m;A_i++){
            for(int A_j = 0; A_j < k;A_j++){
                C[A_i*n+B_i]+=A[A_i*k+A_j]*B[A_j*n+B_i];
            }
        }
    }
}
```

同样编译运行：

<img src="https://gitee.com/e-year/images/raw/master/img/202403191949584.png" alt="image-20240319194925996" style="zoom:67%;" />

##### 4. 编译优化

对C语言代码的编译指令加入``-Ofast``，如``gcc -Ofast matrix.c -o opt_matrix.out``

<img src="https://gitee.com/e-year/images/raw/master/img/202403191951275.png" alt="image-20240319195129768" style="zoom:67%;" />

##### 5. 循环展开

对C语言代码的编译指令加入``-funroll-loops``，如``gcc -funroll-loops matrix.c -o roll_matrix.out``

<img src="https://gitee.com/e-year/images/raw/master/img/202403191952769.png" alt="image-20240319195225364" style="zoom:67%;" />

##### 6. Intel MKL

直接调用``cblas_dgemm``函数对A、B矩阵相乘得到C

```c
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
```

编译运行

```shell
. /opt/intel/bin/compilervars.sh intel64
gcc matrix_mkl.c -lmkl_rt -o matrix_mkl.out
./matrix_mkl.out
```

<img src="https://gitee.com/e-year/images/raw/master/img/202403191953842.png" alt="image-20240319195323447" style="zoom:67%;" />

### 性能对比

- 浮点性能 = 浮点运算次数/运行时间
- 峰值性能 = 核数 × 时钟频率 = 35.199968

| 版本 | 实现描述     | 运行时间（sec） | 相对加速比 | 绝对加速比 | 浮点性能（GFLOPS） | 峰值性能百分比 |
| ---- | ------------ | --------------- | ---------- | ---------- | ------------------ | -------------- |
| 1    | C/C++        | 6.8357          | 1          | 1          | 0.3142             | 0.0089         |
| 2    | Python       | 711.6908        | 0.01       | 0.01       | 0.003              | 8.5e-5         |
| 3    | 调整循环顺序 | 35.8837         | 19.83      | 0.1905     | 0.0598             | 0.0016         |
| 4    | 编译优化     | 1.8046          | 19.88      | 3.7879     | 1.1900             | 0.0338         |
| 5    | 循环展开     | 6.7808          | 0.2661     | 1.0008     | 0.3167             | 0.0090         |
| 6    | Intel MKL    | 0.5847          | 11.60      | 11.6910    | 3.6728             | 0.1043         |

**总结：**通过对比各个版本的性能，可知C语言在串行矩阵乘法上的性能远大于没有调用numpy的Python版本，在其他优化版本中，编译优化以及是调用MKL库的方式能够明显加速串行矩阵乘法。