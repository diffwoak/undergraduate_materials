pequation.c ——一元二次方程求解
	—main 		主函数
	—double_b 	计算b^2
	—four_ac		计算4*a*c
	—sqrt_bac	计算b^2-4*a*c开方
	—two_a		计算2*a

编译运行
gcc -g -Wall -o pe pequation.c -lpthread -lm
./pe

Montecarlo.c ——蒙特卡洛方法求pi的近似值
	—main 主函数
	—partial_count线程执行统计数据落在圆内的数量

编译运行
gcc -g -Wall -o pm Montecarlo.c -lpthread
./pm 4