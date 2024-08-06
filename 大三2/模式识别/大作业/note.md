## 模式识别大作业笔记

### 作业要求

- **选题**：开放词汇视觉感知的下游应用

围绕其中一个或多个展开综述：

**场景理解**：视觉语言导航、场景图生成、3D实例分割

视频理解：视频实例分割、多目标跟踪

文本动作生成

- **注意**

1. 问题的描述、技术难点、各相关论文的研究动机、多篇论文技术要点的层次逻辑关系、技术总结的准确性、详尽性等
2. 其它相关工作介绍和调研
3. 实验结果展示、已有工作总结及未来展望介绍等

- **提交**

1. 7 页 latex 英文论文格式进行撰写，加上参考文献不多于 8 页
2. 海报按给定模板制作

### 作业过程





step

1. 阅读ppt

> - 场景图生成
>
> 旨在将图像解析成一个图表示，**描述物体实体及其之间的关系**
>
> ➢ 传统SSG：封闭世界环境，可能导致场景表示不完整，和与下游任务之间的领域差距。
>
> ➢ PGSG：提供了一个统一的框架，可以直接从图像中生成具有新谓词的场景图，并进行VL任务。
>
> **PGSG基本思想：**
>
> ➢ **给场景图序列提示，使用视觉语言模型生成场景图序列**
>
> ➢ **从场景图序列的自然语言描述中解析三元组**
>
> ➢ **实体定位：**使用编码器-解码器预测实体的边界框
>
> ➢ **类别转换：**预测从词汇空间转换为类别空间
>
> **训练与推理**
>
> ➢ **训练**
>
> ​	➢ **多任务损失：标记下一个标记预测语言建模损失** 𝑳𝒍𝒎 **和实体定位模块的损失**𝑳𝒑𝒐𝒔
>
> ➢ **推理**
>
> ​	➢ **通过VLMs生成场景图序列**
>
> ​	➢ **从场景图序列中解析关系三元组**
>
> ​	➢ **实体定位与类别转换**
>
> **性能表现**
>
> ➢ **SGG基准测试**
>
> ➢ **三个基准测试：PSG， VG和OIv6**
>
> ➢ **三个评估协议SGDet、PCls 和SGCls**
>
> ➢ **随机选择50%的谓词类别作为开放词汇谓词SGG的新类别。**
>
> ➢ **下游视觉语言任务**
>
> ➢ **视觉问答**
>
> ➢ **图像标题生成**
>
> ➢ **视觉定位**
>
> 
>
> - 3D实例分割
>
> 旨在**预测3D对象实例掩码及其对象类别**
>
> ❑传统3D实例分割：通常只能识别预定义的、在训练数据集中进行了注释的对象类别，存在局限性。
>
> ❑OpenMask3D：零样本开放词汇3D实例分割方法，具有超出预定义概念的实例分割能力
>
> **OpenMask3D基本思想：**
>
> ➢ **利用了一个预训练的3D实例分割模型的掩码模块，计算类别无关的实例掩码proposal**
>
> ➢ **掩码特征计算：为每个预测实例掩码，通过CLIP计算一个与任务无关的特征表示**
>
> ➢ **计算得到的特征用于概念查询：3D实例分割**
>
> **模型结果分析：**
>
> ➢ **使用ScanNet200和Replica数据集进行实验**
>
> ➢ **OpenMask3D与其他开放词汇方法相比有显著优势，在尾部类别上的性能明显优于其他开放词汇方法。**
>
> 
>
> - 视觉语言导航
>
> 旨在构建智能代理程序，能够**按照自然语言指令在未知环境中导航**。
>
> ➢ 迭代式视觉与语言导航（IVLN）允许代理程序利用跨指令记忆以实现更好的导航性能。
>
> ➢ 传统IVLN的挑战：
>
> ​	➢ 跨指令记忆缺乏明确的监督信息以处理复杂的环境和指令
>
> ​	➢ 结构化跨指令记忆以便于模型利用
>
> ➢ OVER-NAV: 使用开放词汇检测，结构化场景记忆
>
> **OVER-NAV基本思想：**
>
> ➢ **使用LLM提取指令中的关键概念**
>
> ➢ **使用开放词汇检测器对记忆中的图像进行开放词汇检测**
>
> ➢ **构建场景的图结构化表示，将开放词汇检测结果保存在节点上**
>
> ➢ **实时计算场景记忆图信息，以文本模态送入VLN代理程序**
>
> **性能表现**
>
> ➢ **分别在离散场景和连续场景测试性能**
>
> ➢ **以HAMT和MAP-CMA两种基础模型为例验证方法有效性**

2. 浅看文献

3. 求助gpt（gpt翻译总结，书写中文格式的草稿，再由gpt转为英文）



4. 查看latex格式如何使用，分为





研究背景/研究目的与意义；Abstract、 Introduction

研究现状；Literature Review、Technical Approaches、Application Cases

评述（总结）Discussion、Conclusion

参考文献 References

**References**:

[1] Ganlong Zhao, Guanbin Li, Weikai Chen, and Yizhou Yu. OVER-NAV: Elevating iterative vision-and-language navigation with open-vocabulary detection and structured representation. arXiv preprint arXiv:2403.17334, 2024.

[2] Aya Takmaz, Jonas Schult, Elisabetta Fedele, Robert W. Sumner, Marc Pollefeys, Federico Tombari, and Francis Engelmann. OpenMask3D: Open-vocabulary 3D instance segmentation. arXiv preprint arXiv:2306.13631, 2023.

[3] Rongjie Li, Songyang Zhang, Dahua Lin, Kai Chen, and Xuming He. From pixels to graphs: Open-vocabulary scene graph generation with vision-language models. arXiv preprint arXiv:2404.00906, 2024.

