## Project design

实现细粒度图像分类，数据集``CUB 200 Bird``、``Stanford Dogs``

**4.20**: finish the first 3 points;

1. 提升模型性能的训练技巧
2. 微调预训练模型
3. 用生成模型进行数据增强，如GAN、diffusion models



**5.18**: finish the first 5 points;

4. ViT model 对比CNN，如何有效使用ViT
5. 探索VLM（vision-language model）

**6.15**: finish all must-do points;

6. 模型可解释性：可视化，分析正确和不正确的预测
7. 评估模型的鲁棒性：输入对抗性例子
   1. 提升模型鲁棒性
   2. 得到轻量级模型
8. 经验评价：在不同数据集，超参数的敏感度

**6.29**: submit report with source code

注意：完成第一个point后就要开始写report，每个小组成员都提供自己的源码以及running files

### 实验过程

#### 0. 读代码

```=-
main.py -- trainer.py
		 - config.py				# 配置文件
		 - data_utils/csv_reader		# 读取csv文件
		 - converter/common_utils --save_as_hdf5 # 保存为hdf5中间结果 
trainer.py -- utils.py --dfs_remove_weight
		    - data_utils/transform		# 数据转换和数据曾倩法国
		    - data_utils/data_loader	# 数据读取
		    - import model/resnet.py
		    - import model/vision_transformer.py
config.py -- utils.py 		# 获取path和list函数

		 
other:
	main_GAN.py -- data_utils/transform.py
				 - data_utils/data_loader.py -- DataGenerator
				 - model/gan.py
				 - config.py
				 - data_utils/csv_reader -- csv_reader_single
	main_interpretability.py -- data_utils/transform.py
							  - converter/common_utils
							  - analysis/analysis_tools
							  - config.py
	main_VLM.py
	make_csv.py
	model/gan.py
	model/resnet.py
	model/vision_transformer.py
	converter/tools.py -- converter/common_utils
	analysis/analysis_tools.py -- converter/common_utils
	analysis/statistic_result.py 
	analysis/result.py				
```

[🤗 PEFT - 【布客】huggingface 中文翻译 (apachecn.org)](https://huggingface.apachecn.org/docs/peft/)



[CUB_200_2011 数据集预处理批量 crop 裁剪 + split 划分 python 实现_数据集crop-CSDN博客](https://blog.csdn.net/weixin_43667077/article/details/104809196)









### 训练技巧 -- from ppt

2. 卷积神经网络CNN，使用哪个激活函数，多个卷积核，high-level filter（stride，pooling）

   <img src="https://gitee.com/e-year/images/raw/master/img/202403241445184.png" alt="image-20240324144524567" style="zoom: 67%;" />

   3. 最小化损失函数，反向传播BP，批量梯度下降、随机梯度下降、**小批量梯度下降**，Momentum；过拟合、泛化能力：提前停止训练，正则化，dropout；增强方法：平移旋转缩放剪切变形颜色；集成模型ensemble model（不同初始化参数，结构）

      <img src="https://gitee.com/e-year/images/raw/master/img/202403241511887.png" alt="image-20240324151132730" style="zoom:50%;" />

      4. 一个梯度取决于之前的参数w以及激活函数的导数和wx的x

         梯度爆炸：初始化令前二者小于等于1，重新归一化w，重新缩放x<=1，

         梯度消失：choose RELU，初始化w高斯分布或平均分布

         跨层的信号不会增缩，或信号的方差不会增缩，且反向传播信号的方差也不变

         Xavier's method

         Kaiming's method(activation is ReLU)

         mini-batch的每批都有不同的分布，导致无法收敛，解决：batch normalization（BN），能够加速测试集正确率提升速度

         BN在batch size很小的时候不起很好作用，因此来到Group normalization（GN），缺点：随着layer变深会有更大的错误率

         ResNet