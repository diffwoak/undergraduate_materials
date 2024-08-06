


matrix.cu	——矩阵转置
	—main主函数
	—cpuSecond 获得当前时间（us）
	—initialize 随机初始化矩阵
	—show 展示矩阵元素
	—mat_multi 矩阵相乘函数 （初始版本：只是用全局内存）
	—mat_multi_shared 矩阵相乘函数 （使用共享内存）
	—mat_multi_shared_v1矩阵相乘函数 （使用共享内存并指定数据划分方式）
	—main 主函数
编译运行
nvcc matMulti.cu -o mat
./mat