#  迁移学习

迁移学习指的是首先将模型在额外数据上进行预训练，再将预训练模型“迁移"到现有问题的技术。 迁移学习技术有助于提升模型在小数据集上的表现，使用他人发布的预训练模型还可以减少训练的时间和资源消耗，在以transformer为代表的大模型领域被广泛应用。根据如何将预训练模型“迁移“到现有问题上，以下是几个迁移学习的思想

transfer Learning:冻结预训练模型的全部卷积层(主要是提取特征的层，消耗资源较多)，只训练自己定制的全连接层(主要是分类器层，消耗资源较少)

ExtractFeature Vector:先计算出预训练模型的卷积层对所有训练和测试数据的特征向量，然后抛开预训练模型，只训练自己,制的简配版全连接网络(行为上和上一条类似)

Fine-tune:冻结预训练模型的部分层(通常是靠近输入的多数卷积层)，训练剩下的层(通常是靠近输出的部分卷积层和全连接层)(TODO为什么是这样)。

目前Fine-tune是最常用的迁移学习技术，称为微调，如何对预训练模型进行微调，是一个值得深入研究的话题。本项目代码的resnet.py和vision transformer.py原样拷贝了torchvision库的相关内容，TODO尝试修改resnet,py和vision_transformer.py，实现

下载并使用预训练模型的权重，并设计微调策略，冻结部分层训练其他层



1. 迁移学习

Transfer learning专注于存储已有问题的解决模型，并将其利用在其他不同但相关问题上。一个典型场景是，当原始域和目标域的数据分布不同但存在一定的相关性或相似性时，我们可以通过迁移学习方法来提升在目标域上的学习效果。

定义

迁移学习是由域和任务定义的。“域”由特征空间和边缘概率分布构成，特征空间表示可能的输入数据的集合，而边缘概率分布则描述这些输入数据的分布情况。给定一个域\[{\mathcal {D}}=\{{\mathcal {X}},P(X)\}\]，一个任务由标签空间和目标预测函数两部分组成，目标预测函数用于预测输入数据对应的标签。任务\[{\mathcal {T}}=\{{\mathcal {Y}},f(x)\}\]是通过学习包含训练样本对的训练数据得到的。

给定原始域\[{\mathcal {D}}_{S}\]及其任务\[{\mathcal {T}}_{S}\]，目标域\[{\mathcal {D}}_{T}\]及其任务\[{\mathcal {T}}_{T}\]，满足原始域和目标域的特征空间或任务不完全相同。迁移学习的目标就是利用原始域和任务的知识，来帮助学习目标域的目标预测函数\[f_{T}(\cdot )\]，也就是在目标域上利用原始域的信息来改善学习性能或泛化能力，而不是从零开始学习。

可行性

迁移学习可行的原因在于深度学习模型属于层叠架构，在不同层学习不同的特征，最后连接到最终层，一般为一个全连接层，以得到最终输出。这样的层叠架构让我们可以利用预训练网络，去其最终层，将其作为固定的特征提取器，用于其他任务。关键在于只使用预训练模型的权重层提取特征，在为新任务训练新数据的时候不更新这些预训练层，这也就是基础的transfer learning，冻结预训练模型的全部卷积层(主要是提取特征的层，消耗资源较多)，只训练自己定制的全连接层(主要是分类器层，消耗资源较少)。也存在其他的迁移方式，比如ExtractFeature Vector，先计算出预训练模型的卷积层对所有训练和测试数据的特征向量，然后抛开预训练模型，只训练自己,制的简配版全连接网络；以及最常见的Fine-tune，冻结预训练模型的部分层(通常是靠近输入的多数卷积层)，训练剩下的层(通常是靠近输出的部分卷积层和全连接层)，在本文中也是主要使用Fine-tune对预训练模型展开训练，加载了Resnet以及ViT预训练模型进行微调训练，并对比从头开始训练一个网络的结果，在训练效率和最终准确率上都取得了良好效果。

迁移学习的作用

 迁移学习技术有助于提升模型在小数据集上的表现，在当前transformer为代表的大模型领域被广泛应用的情况下，更需要迁移学习来帮助在小数据集上表现不良的大模型，

Enhanced performance through leveraging pre-trained models and fine-tuning, especially beneficial with limited data. By starting from pre-trained models, it can reduce training time and avoid the need to train from scratch on large datasets. Improved generalization as pre-trained models capture general patterns and features, aiding adaptation to new tasks.

Versatility and flexibility across different architectures and domains, making it a widely applicable technique in machine learning. 

[迁移学习全面指南：概念、应用、优势、挑战 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/51620530)



---

Transfer learning focuses on leveraging pre-existing solution models and applying them to different but related problems. A typical scenario is when the data distribution of the source domain and target domain differ but exhibit some correlation or similarity. In such cases, transfer learning methods can enhance learning performance on the target domain.

### Definition
Transfer learning is defined by domains and tasks. A "domain" consists of a feature space and marginal probability distribution, where the feature space represents the set of possible input data, and the marginal probability distribution describes the distribution of these input data. Given a domain \(\mathcal{D}=\{\mathcal{X}, P(X)\}\), a task comprises a label space and a target prediction function, with the target prediction function used for predicting labels corresponding to input data. A task \(\mathcal{T}=\{\mathcal{Y}, f(x)\}\) is learned from training data containing pairs of input and labels.

Given a source domain \(\mathcal{D}_S\) with its task \(\mathcal{T}_S\) and a target domain \(\mathcal{D}_T\) with its task \(\mathcal{T}_T\), where the feature space or tasks between the source and target domains are not entirely identical, the goal of transfer learning is to utilize knowledge from the source domain and task to aid learning the target prediction function \(f_T(\cdot)\). This involves leveraging information from the source domain to improve learning performance or generalization on the target domain, rather than starting from scratch.

### Feasibility
Transfer learning is feasible due to the layered architecture of deep learning models, where different layers learn different features and are connected to a final layer, often a fully connected layer, to obtain the final output. This architecture allows us to utilize pre-trained networks, remove their final layer, and use them as fixed feature extractors for other tasks. The key is to use only the weight layers of the pre-trained model to extract features and not update these pre-trained layers when training new data for a new task. This is the basis of transfer learning, where we freeze the entire convolutional layers of the pre-trained model (which mainly extract features and consume more resources) and only train our custom fully connected layers (which mainly include the classifier layers and consume fewer resources). Other transfer methods also exist, such as Extract Feature Vector, which calculates the feature vectors of the pre-trained model's convolutional layers for all training and test data and then trains a simplified version of the fully connected network separately. The most common method is Fine-tune, where we freeze part of the pre-trained model's layers (usually the majority of convolutional layers near the input) and train the remaining layers (usually the part of convolutional layers near the output and fully connected layers). In this paper, Fine-tune is mainly used to train pre-trained models. We perform fine-tuning training on ResNet and ViT pre-trained models and compare the results with training a network from scratch. We achieve good results in terms of training efficiency and final accuracy.

Transfer learning technology helps improve model performance on small datasets, which is especially crucial in the current context of large models represented by transformers, where transfer learning is widely applied to assist large models that perform poorly on small datasets.By starting with the pre-trained model and taking advantage of the generalization obtained from the pre-training, we can avoid the need to train from scratch on large datasets and reduce training time effectively.

---

