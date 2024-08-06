matConv.cu——直接卷积
	—main主函数
	—cpuSecond 获得当前时间（us）
	—loadMat 加载输入矩阵、kernel、output
	—show 展示矩阵元素
	—directConv2D 直接卷积 核函数

编译运行
nvcc matConv.cu -o con
./con

matIm2col.cu——使用im2col方法卷积
	—main主函数
	—cpuSecond 获得当前时间（us）
	—loadMat 加载输入矩阵、kernel、output
	—show 展示矩阵元素
	—imtocol 将input转为待计算矩阵 核函数
	—mat_multi 矩阵乘积 核函数
编译运行
nvcc matIm2col.cu -o col
./col

matcuDNN.cu——使用cuDNN库方法卷积
	—main主函数
	—cpuSecond 获得当前时间（us）
	—loadMat 加载输入矩阵、kernel、output
	—show 展示矩阵元素
编译运行
export LD_LIBRARY_PATH=/opt/conda/lib:SLD_LIBRARY_PATH
nvcc matcuDNN.cu -I/opt/conda/include -L/opt/conda/lib -lcudnn -o cud
./cud

testConv.py ——生成input、kernel输入，以及output输出用于比对正确性