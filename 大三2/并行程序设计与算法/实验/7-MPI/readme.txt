parallel.h	——调用创建多个Pthreads线程
	—parallel_for函数
生成动态链接库
gcc parallel.h -fPIC -shared -o libpa.so -lpthread



matrix.c	——矩阵乘法函数
	—main主函数
	—matrix_multiple_thread 矩阵乘法线程函数
编译运行
gcc -g -Wall matrix.c -L. -lpa -o mat -lpthread
./mat 4



fft_parallel.cpp
	——main 主函数
	——cfft2：FFT函数，主进程调用
	——cfft2_sub：子进程辅助主进程的FFT函数
	——step：供划分数据后进程调用的FFT步骤
	——reorderArray：FFT之前的数据重组
	——其他函数：保留serial版本未修改过
编译运行：
mpicxx -g -Wall -o par fft_parallel.cpp
mpirun -np 4 ./par


