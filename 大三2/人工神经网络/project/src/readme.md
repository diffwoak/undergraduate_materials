项目文件运行：

```bash
# Resnet和ViT模型 运行main.py,修改配置config.py,数据集未放入项目
python main.py -m train
python main.py -m inf
# GAN模型 运行main_GAN.py,配置config_gan.py,数据集未放入项目
python main_GAN.py
# DCGAN模型 运行main_DCGAN.py,配置直接于该文件中修改全局变量,数据集未放入项目
python main_DCGAN.py
# ACGAN模型,数据集未放入项目
cd ACGAN
python main.py --dataset CUB_200  --dataroot ../datasets/CUB_200_2011/CUB_200_2011/images --cuda
# VLM 模型 使用clip-vit-base-patch32,模型均未放入项目
python main_VLM.py
# CAM 注：mid_feature中未保存中间结果,由另一小组成员记录结果
python main_interpretability.py
# 对抗攻击,数据集未放入项目
python adversarial_attack.py
```

项目结果如下：

```
|-- analysis/ 模型结果输出的文件夹，以及数据统计工具类
    |-- result/ 最终结果输出文件夹
        |-- xxx/ 按版本号输出的文件夹
            |-- foldk.csv 第k折交叉验证的预测结果
            |-- foldk_report.csv 第k折交叉验证的分类报告
            |-- foldk_fc_weight.npy 第k折交叉验证的全连接层权重
    |-- mid_feature/ 钩子（hook）获取的中间结果（某层的输入输出特征）输出文件夹
        |-- xxx/ 按版本号输出的文件夹
            |-- foldk/ 按第k折交叉验证输出的文件夹
                |-- xxx 逐个输出测试集样本的中间结果
    |-- analysis_tools.py 绘出混淆矩阵、roc曲线、计算CAM、保存热力图的类
    |-- statistic_result.py 统计评价指标和auc的类
|-- ckpt/ 模型参数输出的文件夹
    |-- xxx/ 按版本号输出的文件夹
        |-- foldk/ 按第k折交叉验证输出的文件夹
            |-- epoch=xxx.pth 模型参数输出
|-- clip-vit-base-patch32/ VLM调用模型
|-- converter/ 文件读取的工具类的文件夹
|-- csv_file/ csv格式的数据集目录的文件夹
|-- data_utils/ 数据读取的工具类的文件夹
    |-- data_loader.py 数据读取
    |-- transform.py 数据转换和数据增强
|-- datasets/ 数据集的文件夹
    |-- CUB_200_2011/ CUB 200 2011数据集	未提交数据集
	|-- Stanford_Dogs/ Stanford Dogs数据集	未提交数据集
|-- gan_dataset/ 生成模型生成的数据集的文件夹
|-- log/ 模型运行的过程数据输出的文件夹
    |-- xxx/ 按版本号输出的文件夹
        |-- foldk/ 按第k折交叉验证输出的文件夹
            |-- data/ 过程数据输出文件夹
            |-- events.xxxxx 命令行输出结果
|--res_pic/
	|-- xxx/ 按版本号输出的文件夹
	|-- {model}_{fold}_accuracy.jpg 不同模型准确率折线图
	|-- {model}_{fold}_Loss.jpg		不同模型损失折线图
|--ACGAN	ACGAN模型文件目录
	|--main.py SNGAN模型的入口
|-- model/ 模型文件夹
|-- main.py 项目模型入口
|-- config.py 模型配置 main.py调用
|-- config_gan.py 模型配置 main_GAN.py调用
|-- LICENSE 权利声明
|-- main_GAN.py 简单的GAN模型的入口
|-- main_DCGAN.py DCGAN模型的入口
|-- main_interpretability.py CAM保存热力图的入口
|-- main_VLM.py VLM模型的入口
|-- readme.md 	项目说明
|-- requirements.txt 依赖库
|-- make_csv.py 制作CUB_200_2011的csv格式的数据集目录的函数
|-- make_stanford_dogs_csv.py 制作stanford_dogs的csv格式的数据集
|-- trainer.py 训练器，训练过程主要代码
|-- compute_mean_and_std.py 计算数据集均值和标准差的函数
|-- adversarial_attack.py 对抗攻击函数
|-- utils.py 工具类
```

