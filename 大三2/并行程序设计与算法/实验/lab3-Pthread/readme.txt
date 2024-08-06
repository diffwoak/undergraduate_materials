pmatrix.c ——并行矩阵乘法
	—main 主函数
	—matrix_multiple 并行矩阵乘法入口
	—matrix_multiple_thread 线程执行的部分矩阵乘法操作

编译运行
gcc -g -Wall -o pth pmatrix.c -lpthread
./pth 4

psum.c ——并行数组总和
	—main 主函数
	—partial_sum 线程执行部分数组加和操作

编译运行
gcc -g -Wall -o ps psum.c -lpthread
./ps 4