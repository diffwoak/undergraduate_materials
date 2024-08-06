helloworld.cu	——CUDA Hello World
		—helloFromGPU函数 输出 GPU 的 hello world
		—main 主函数 输出host 的hello world

nvcc helloworld.cu -o hello
./hello



matrix.cu	——矩阵转置
	—main主函数
	—cpuSecond 获得当前时间（us）
	—initialize 随机初始化矩阵
	—mat_transpose 矩阵转置函数 （初始版本：只是用全局内存）
	—mat_transpose_share 矩阵转置函数 （使用共享内存并间隔划分线程）
	—mat_transpose_share_v1 矩阵转置函数 （使用共享内存并连续划分线程）
	—main 主函数
编译运行
nvcc matrixT.cu -o mat
./mat