## Lab7 - MPI并行应用

### 实验要求

使用MPI对快速傅里叶变换进行并行化。

**问题描述**：阅读参考文献中的串行傅里叶变换代码(fft_serial.cpp)，并使用MPI对其进行并行化。

**要求**：

1. 并行化：使用MPI多进程对fft_serial.cpp进行并行化。为适应MPI的消息传递机制，可能需要对fft_serial代码进行一定调整。

2. 优化：使用MPI_Pack/MPI-Unpack或MPI_Type_create_struct对数据重组后进行消息传递。

3. 分析：

   a) 改变并行规模（进程数）及问题规模（N），分析程序的并行性能；

   b) 通过实验对比，分析数据打包对于并行程序性能的影响；

   c) 使用Valgrind massif工具集采集并分析并行程序的内存消耗。

   注意Valgrind命令中增加--stacks=yes 参数采集程序运行栈内内存消耗。Valgrind massif输出日志（massif.out.pid）经过ms_print打印。工具参考：[Valgrind](https://valgrind.org/docs/manual/ms-manual.html)


### 实验过程

##### 1. 实现思路

串行傅里叶变换代码中能够并行化的部分是进入函数``cfft2``之后，根据问题规模N的大小调用一次或多次`step`函数，因此实现的思路是，使用多个进程划分``step``函数任务并行执行，进程间通讯则考虑``step``之前需要主进程将计算量分发到子进程，``step``之后主进程接收子进程的计算结果。

关于进程数，里面涉及一个步长单元``mj``，如果一直使用多个进程执行``step``，每个进程划分到的范围可能小于mj，也可能包含多个mj，导致计算结果范围也不在划分范围内，实现复杂度较高，因此根据mj大小，随着mj的增大，会有减少进程量的操作，使每个进程划分范围至少存在一个完整的步长``mj``。

##### 2. 代码实现

``reorderArray``数据重组，在执行傅里叶变换之前，根据FFT过程中数组的元素结合顺序，例如[0, 1, 2, 3, 4, 5, 6, 7]中，第0个元素和第4个元素配对成04，第1个元素与第3个元素配对成13，最终会呈现的顺序其实是[0, 4, 2, 6, 1, 5, 3, 7]，配对如下：

``` 
0 4 2 6 1 5 3 7
| / | / | / | /
04  26  15  37
 | /    | /
 0426  1537
   |  /
 04261537
```

如此重排能够使数据划分到不同进程时数组元素连续，将原本需要发送两次的数组元素一次发送，降低通讯成本

```C++
void reorderArray(double* arr, int n) {/*在每次调用cfft2前使用该函数*/
    double* tmp1 = new double[n * 2];
    double* tmp2 = new double[n * 2];
    for (int i = 0; i < n * 2; i++) {
        tmp2[i] = arr[i];
    }
    for (int mj = 1; mj < n / 2; mj = mj * 2) {
        int index = 0;
        for (int i = 0; i < (n / 2) / mj; i++) {
            for (int j = 0; j < mj; j++) {
                tmp1[index++] = tmp2[(i * mj + j) * 2];
                tmp1[index++] = tmp2[(i * mj + j) * 2 + 1];
            }
            for (int j = 0; j < mj; j++) {
                tmp1[index++] = tmp2[(n / 2 + i * mj + j) * 2];
                tmp1[index++] = tmp2[(n / 2 + i * mj + j) * 2 + 1];
            }
        }
        for (int i = 0; i < n * 2; i++) {
            tmp2[i] = tmp1[i];
        }
    }
    for (int i = 0; i < n * 2; i++) {
        arr[i] = tmp2[i];
    }
    delete[] tmp1;
    delete[] tmp2;
}
```

``step``函数，

```C++
void step ( int n, int mj, double a[], double b[], double c[],double d[], 
           double w[], double sgn,int begin, int end ){
    /*传入begin、end参数表示每个进程的划分范围*/
  double ambr,ambu;
  int j,jw;
  int k;
  double wjw[2];
  int start = 0; // 表示每个子进程中a,b,c,d的索引
  for (j = begin / mj; j < end / mj; j++)
  {
    jw = j * mj;
    wjw[0] = w[jw*2+0]; // cos
    wjw[1] = w[jw*2+1]; // sin
    if ( sgn < 0.0 ) {
      wjw[1] = - wjw[1];
    }
    // a、b来自同一数组的不同位置，c、d来自同一数组的不同位置
    for (k = 0; k < mj; k++)// 在一个mj中连续取a、b值，连续赋c、d值
    {
        c[(start) * 2 + 0] = a[(start) * 2 + 0] + b[(start) * 2 + 0];
        c[(start) * 2 + 1] = a[(start) * 2 + 1] + b[(start) * 2 + 1];

        ambr = a[(start) * 2 + 0] - b[(start) * 2 + 0];
        ambu = a[(start) * 2 + 1] - b[(start) * 2 + 1];

        d[(start) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
        d[(start) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        start++;
    }
   	// 数组中如此存储 ababab，因此下一个mj范围的索引需要+=mj
    start += mj;
  }
  return;
}
```

``cff2``函数：主进程执行

```C++
void cfft2(int n, double x[], double y[], double w[], double sgn, int comm_sz)
{/* comm_sz在这表示实际执行任务的进程数 */
    int j, m, mj;
    int tgle;
    int pro_nums = comm_sz;
    int counts = (n / 2) / pro_nums;	// 每个进程划分量
    int begin = 0, end = counts;		// 0号进程执行的起始范围
    m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
    mj = 1;
    tgle = 1;
    for (int i = 1; i < pro_nums; i++) {// 1.分发数据到各个进程
        MPI_Send(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }
    step(n, mj, &x[0], &x[mj * 2], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
    for (int i = 1; i < pro_nums; i++) {// 2.收集各个进程计算结果
        MPI_Recv(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (n == 2)return;
    for ( j = 0; j < m - 2; j++ )
    {
        mj = mj * 2;
        // 根据mj重新划分,counts翻倍,pro_nums减半
        pro_nums = min((n / 2) / mj, pro_nums);
        counts = (n / 2) / pro_nums;
        end = counts;
        if ( tgle )
        {
            for (int i = 1; i < pro_nums; i++) {// 1.分发数据到各个进程
                MPI_Send(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }
            step(n, mj, &y[0], &y[mj * 2], &x[0], &x[mj * 2 + 0], w, sgn, begin, end);
            tgle = 0;
            for (int i = 1; i < pro_nums; i++) {// 2.收集各个进程计算结果
                MPI_Recv(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } 
        }
        else
        {   
            for (int i = 1; i < pro_nums; i++) {// 1.分发数据到各个进程
                MPI_Send(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }
            step(n, mj, &x[0], &x[mj * 2], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
            tgle = 1;
            for (int i = 1; i < pro_nums; i ++) {// 2.收集各个进程计算结果
                MPI_Recv(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    if ( tgle ){
        ccopy ( n, y, x );
    }
    mj = n / 2; 
    end = mj;
    // 只需使用主进程
    step(n, mj, &x[0], &x[n], &y[0], &y[n], w, sgn, begin, end);
    return;
}
```

``cff2_sub``函数：子进程执行

```C++
void cfft2_sub(int n, double w[], double sgn,int my_rank, int comm_sz)
{	/*	采用与cfft2主函数相似的逻辑构造，使主函数每个send recv都有对应的子进程recv send
		w在进入函数前已经广播到每个进程*/
    int j;
    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    int pro_nums = comm_sz;
    int counts = (n / 2) / pro_nums;
    int begin = my_rank * counts;
    int end = begin + counts;
    // 数组x，y随着mj大小变换而变化，每次step需要重新分配空间（或直接分配一个足够大的空间）
    double* x, * y;
    x = new double[2 * 2 * counts];
    y = new double[2 * 2 * counts];
    // 1.接收主进程的计算前数据
    MPI_Recv(&x[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    step(n, mj, &x[0], &x[mj * 2 + 0], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
    // 2.发还主进程的计算后数据
    MPI_Ssend(&y[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    delete[] x;
    delete[] y;
    if (n == 2)return;
    for (j = 0; j < m - 2; j++)
    {
        mj = mj * 2;
        // 根据mj重新划分,counts翻倍,pro_nums减半
        pro_nums = min((n / 2) / mj, pro_nums);
        counts = (n / 2) / pro_nums;
        begin = my_rank * counts;
        end = begin + counts;
        if (my_rank < pro_nums) {
            x = new double[2 * 2 * counts];
            y = new double[2 * 2 * counts];
            // 1.接收主进程的计算前数据
            MPI_Recv(&x[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step(n, mj, &x[0 * 2 + 0], &x[mj * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn, begin, end);
            // 2.发还主进程的计算后数据
            MPI_Ssend(&y[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            delete[] x;
            delete[] y;
        }
    }
    return;
}
```

##### 3. 优化

实验要求使用数据重组后进行消息传递(MPI_Pack/MPI-Unpack或MPI_Type_create_struct)，但是依照我已经实现的代码（使用``reorderArray``对即将FFT的数据预处理），在每次传播时只需要发送一段连续的数组元素，即只需调用一次``MPI_Send``，在此基础上进行MPI_Pack/MPI-Unpack或MPI_Type_create_struct并没有实际比对意义。

如果使用MPI_Pack，需要对数组x的分段元素进行多次打包，再依次传送，这对原本连续的数组元素打包并无太大作用；如果使用MPI_Type_create_struct，那么只是在一个结构体中存储一个指针，反而在定义存储结构体上有更多开销。二者本质上都是用来传播多个数据项的复杂数据结构的方式，在我已经实现的并行化代码基础上完全没有优化的意义和必要，可将实现的``reorderArray``数据重组函数看作相似的优化处理，在此不做更多处理。

##### 4. 运行结果

编译运行指令：

```shell
# 串行
g++ fft_serial.cpp -o ser
./ser
# 并行
mpicxx -g -Wall -o par fft_parallel.cpp
mpirun -np 4 ./par
```

在虚拟机上安装Valgrind

```
wget https://sourceware.org/pub/valgrind/valgrind-3.19.0.tar.bz2
tar -jxvf valgrind-3.19.0.tar.bz2

sudo apt-get install automake
sudo apt-get install autoconf
 
cd valgrind-3.19.0
./autogen.sh
./configure
make -j4
sudo make install
```

使用Valgrind

```shell
# 串行
valgrind --tool=massif --stacks=yes ./ser
# 并行
mpirun -np 4 valgrind --tool=massif --stacks=yes ./par
```

**串行版本运行结果**

<img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520191747743.png" alt="image-20240520191747743" style="zoom:67%;" />

<img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520191837621.png" alt="image-20240520191837621" style="zoom: 67%;" />

**并行版本运行结果**

此处遇到一些问题，可能是虚拟机配置环境的问题，正常调用检查时未出现错误

<img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520203548899.png" alt="image-20240520203548899" style="zoom:67%;" />

使用mpirun  valgrind运行出错：``cri_init: sigaction() failed: Invalid argument``

<img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520203825171.png" alt="image-20240520203825171" style="zoom:67%;" />

反复搜索各种解决方案尝试修改无果后，反而发现引发另一个错误：正常编译运行该文件会出现每个进程都打印主进程的内容，原本是能正常运行的，出现这个问题不知如何处理，在学校云主机上测试相同文件还是没有问题。

<img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520223258254.png" alt="image-20240520223258254" style="zoom:50%;" /><img src="C:\Users\asus\Desktop\大三下\并行\7-MPI\report.assets\image-20240520223232004.png" alt="image-20240520223232004" style="zoom:50%;" />

**结果分析**

总结上面存在的未解决问题：

1. 使用Valgrind无法正常运行MPI并行文件
2. 调环境过程导致虚拟机上MPI并行文件输出多余内容
3. 结果正确性未找出代码逻辑问题，目前只有N=2，4时的Error是与并行结果一致的

花费很多时间，但上述问题未能在提交作业前得到解决，只能勉强在云主机上进行性能分析，比较不同进程、不同问题规模的Time/Call，以下数据比较来自云主机上实验，因此可能与上文截图有明显差别。

| 进程数\规模 | 8           | 32         | 128        | 512         | 2048        | 8192       |
| ----------- | ----------- | ---------- | ---------- | ----------- | ----------- | ---------- |
| 串行        | 3.9195e-07  | 1.9595e-06 | 9.8345e-06 | 4.831e-05   | 0.00023553  | 0.00114745 |
| 1           | 7.8955e-07  | 4.178e-06  | 2.1394e-05 | 0.000108725 | 0.00052872  | 0.00249685 |
| 2           | 3.34825e-06 | 9.691e-06  | 3.1146e-05 | 0.00014819  | 0.000645515 | 0.0029311  |
| 4           | 6.3177e-06  | 1.5575e-05 | 4.2318e-05 | 0.00014233  | 0.000547    | 0.00247245 |
| 8           | 6.32135e-06 | 2.6461e-05 | 6.1288e-05 | 0.00015594  | 0.000574755 | 0.00245895 |

可以看出并行化后的规模的Time/Call耗时反而比原本串行的长，特别是问题规模较小的时候，并行通信的开销要大于并行带来的计算优势，而在问题规模变大时，如规模=8192一列，同为并行的规模随着进程数变大而耗时变低，才能看出并行化后带来的性能优势。

### 总结

本次实验的难点主要在于理解FFT串行代码并构思并行化的逻辑并重新熟悉之前使用的MPI，由此也容易自己制造很多bug，在改bug的过程耗费了很多时间，导致后面学习使用Valgrind的时间不足，只能完成串行版本的采集程序运行栈内内存消耗，代码还存在需要改正和简化的地方，但因为时间问题，其他作业deadline堆积，只能在最后两天完成这次实验，只能勉强完成基本要求，提交作业之后还需要着手解决上述问题。

