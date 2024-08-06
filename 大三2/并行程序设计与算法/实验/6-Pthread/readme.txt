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



heated_plate_pthread.c——heated_plate_openmp改造为基于Pthreads的并行应用
	——main 主函数
	——initial_value_1	多线程函数：初始化w矩阵左右边界
	——initial_value_2	多线程函数：初始化w矩阵上下边界
	——initialize_solution 多线程函数：初始化w矩阵内部值
	——save_u	多线程函数：迭代过程使用u矩阵存储w矩阵
	——new_w	多线程函数：迭代过程使用u矩阵计算w矩阵
	——update_diff	多线程函数：迭代过程更新my_diff值
编译运行：
gcc -g -Wall heated_plate_pthread.c -L. -lpa -o pth -lpthread
./pth 16


