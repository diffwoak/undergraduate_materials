## resnet模型

在实验的开始，我们会使用Residual Neural Network(ResNet)这一广泛用于图像分类任务的深度学习模型，作为后续实验的比对。ResNet的出现克服了深度网络中的退化问题，成功地训练了比以往模型更深的神经网络，核心在于让网络的每一层不直接学习预期输出，而是学习与输入之间的残差关系。这种网络通过添加“跳跃连接”，即跳过某些网络层的连接来实现身份映射，再与网络层的输出相加合并。这一设计使得拥有几十上百层的深度学习模型可以更易于训练，解决了传统深度神经网络中的梯度消失问题，增加模型深度时还能保持甚至提高准确度，实现了高效、可扩展的深层模型。

**模型原理** 

传统的深度神经网络试图学习输入与输出之间的映射，但是在ResNet中，每个网络层实际上学习的是输入与输出之间的残差映射 ( F(x) = H(x) - x )。然后，这个残差结果与输入 ( x ) 相加，形成 ( H(x) = F(x) + x )。这一机制使得网络更容易学习身份映射，有利于信号在前向传播路径和反向传播路径中的传递，缓解了梯度消失和爆炸的问题。

前向传播

如果第\ell个残差块的输出是第(\ell+1)个残差块的输入（这里假设块与块之间没有激活函数），可以得到：
\begin{align} x_{\ell+1} & = F(x_{\ell}) + x_{\ell}
\end{align}

可以推导出：
\begin{align} x_{L}
& = x_{\ell} + \sum_{i=l}^{L-1} F(x_{i}) \\
\end{align}

这里L表示任意后续残差块的索引（比如处于最末尾的块），\ell代表任意靠前的块对应的索引。通过这个公式也可以看出信号能够从浅层块\ell传递到深层块L。

反向传播

根据上面的前向传播过程，对x_{\ell}进行求导，可以得到：
\begin{align} \frac{\partial \mathcal{E} }{\partial x_{\ell} }
& = \frac{\partial \mathcal{E} }{\partial x_{L} }\frac{\partial x_{L} }{\partial x_{\ell} } \\
& = \frac{\partial \mathcal{E} }{\partial x_{L} } \left( 1 + \frac{\partial }{\partial x_{\ell} } \sum_{i=l}^{L-1} F(x_{i}) \right) \\
& = \frac{\partial \mathcal{E} }{\partial x_{L} }  + \frac{\partial \mathcal{E} }{\partial x_{L} } \frac{\partial }{\partial x_{\ell} } \sum_{i=l}^{L-1} F(x_{i})  \\
\end{align}

这里 \mathcal{E}  是最小化损失函数。以上表明，浅层的梯度计算\frac{\partial \mathcal{E} }{\partial x_{\ell} }总会直接加上一个项\frac{\partial \mathcal{E} }{\partial x_{L} }。因此，由于额外项\frac{\partial \mathcal{E} }{\partial x_{L} }的存在，即使 F(x_{i}) 的梯度很小，总梯度\frac{\partial \mathcal{E} }{\partial x_{\ell} }也不会消失。

**模型架构**

<img src="C:\Users\asus\Desktop\大三下\cvpr2023-author_kit-v1_1-1\latex\img\resnet_1.jpg" alt="image-20240518152810607" style="zoom:33%;" />Residual learning: a building block

一个基础的残差块通常包含以下几个部分：

- 卷积层：用于特征提取。
- 批量归一化（Batch Normalization）：一般跟在卷积层的后面，用于加速训练和改善模型泛化。
- 激活函数：通常使用ReLU。
- 短接连接（Skip Connection）：连接输入和输出![image-20240518160355689](C:\Users\asus\Desktop\大三下\cvpr2023-author_kit-v1_1-1\latex\img\resnet_2.png)

整体架构组成

- 初始卷积层：对输入图像进行一定程度的空间下采样（Spatial Downsampling），减少后续层需要处理的空间维度，从而降低计算复杂度，并获取图像特征，只使用一个较大的卷积核操作能够有效地捕获图像的全局信息
- 残差块组（Residual Blocks Group）：包含多个残差块，每个残差块负责从其前一组中提取的特征中提取更高级的特征，并通过残差链接，每个残差块组能够学习输入与输出之间的复杂非线性映射，同时避免梯度消失和爆炸
- 全局平均池化（Global Average Pooling）：网络的最后一个卷积层，减小维度，降低模型参数和计算量，也能够有效防止模型过拟合，使模型具有更好的泛化能力
- 全连接层：输出固定大小的特征向量，用于分类或者回归任务







---

At the beginning of our experiment, we will use the Residual Neural Network (ResNet), a deep learning model widely used for image classification tasks, as a baseline for subsequent experiments. ResNet addresses the degradation problem in deep networks and successfully trains deeper neural networks than previous models. Its core innovation lies in allowing each layer of the network to learn residual functions relative to the input, rather than directly learning the desired output. This network design includes "skip connections" that bypass certain layers and merge with the layer outputs, facilitating identity mapping. This design makes it easier to train deep learning models with dozens or even hundreds of layers, alleviating the vanishing gradient problem and maintaining or improving accuracy as model depth increases. This enables the creation of efficient, scalable deep models.

**Model Principle**

Traditional deep neural networks aim to learn the mapping between inputs and outputs, but in ResNet, each layer learns the residual mapping ( \( F(x) = H(x) - x \) ). This residual is then added to the input ( \( H(x) = F(x) + x \) ). This mechanism simplifies the learning process for identity mapping and improves signal propagation in both forward and backward paths, mitigating issues like vanishing and exploding gradients.

**Forward Propagation**

If the output of the \(\ell\)-th residual block is the input to the \((\ell+1)\)-th residual block (assuming no activation functions between blocks), we have:
\[ x_{\ell+1} = F(x_{\ell}) + x_{\ell} \]

We can derive:
\[ x_{L} = x_{\ell} + \sum_{i=\ell}^{L-1} F(x_{i}) \]

Here, \(L\) represents the index of any subsequent residual block (e.g., the last block), and \(\ell\) represents the index of any preceding block. This formula shows that signals can propagate from the shallow block \(\ell\) to the deep block \(L\).

**Backward Propagation**

From the forward propagation process, we can derive the gradient with respect to \(x_{\ell}\) as follows:
\[ \frac{\partial \mathcal{E}}{\partial x_{\ell}} = \frac{\partial \mathcal{E}}{\partial x_{L}} \frac{\partial x_{L}}{\partial x_{\ell}} \]
\[ = \frac{\partial \mathcal{E}}{\partial x_{L}} \left( 1 + \frac{\partial}{\partial x_{\ell}} \sum_{i=\ell}^{L-1} F(x_{i}) \right) \]
\[ = \frac{\partial \mathcal{E}}{\partial x_{L}} + \frac{\partial \mathcal{E}}{\partial x_{L}} \frac{\partial}{\partial x_{\ell}} \sum_{i=\ell}^{L-1} F(x_{i}) \]

Here, \(\mathcal{E}\) is the loss function to be minimized. The above equations indicate that the gradient calculation for shallow layers \(\frac{\partial \mathcal{E}}{\partial x_{\ell}}\) always includes an additional term \(\frac{\partial \mathcal{E}}{\partial x_{L}}\). Therefore, due to the presence of this additional term, the overall gradient \(\frac{\partial \mathcal{E}}{\partial x_{\ell}}\) will not vanish even if the gradient of \(F(x_{i})\) is small.

**Model Architecture**

A basic residual block typically consists of the following components:
- Convolutional layers: For feature extraction.
- Batch Normalization: Usually follows the convolutional layers to speed up training and improve model generalization.
- Activation function: Usually ReLU.
- Skip connection: Connects the input to the output.

**Overall Architecture**

- Initial Convolutional Layer: Performs spatial downsampling on the input image, reducing the spatial dimensions for subsequent layers, thereby lowering computational complexity and capturing image features. Using a large convolution kernel can effectively capture global information of the image.
- Residual Block Groups: Contains multiple residual blocks, with each block extracting higher-level features from the previous group's features. Through residual connections, each group can learn complex non-linear mappings between input and output while avoiding gradient vanishing and exploding.
- Global Average Pooling: The final convolutional layer of the network reduces dimensionality, lowers the number of model parameters and computation, and effectively prevents overfitting, leading to better generalization.
- Fully Connected Layer: Outputs a fixed-size feature vector for classification or regression tasks.

---





