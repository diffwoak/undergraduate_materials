import torch
import torchvision.utils as vutils
from network import _netG, _netG_CIFAR10  # 确保正确导入 Generator 模型的定义
from torch.autograd import Variable
import numpy as np

# # 定义生成器模型的一些参数，确保与训练时一致
# ngpu = 1  # 或者根据训练时使用的 GPU 数量来设置
# nz = 110  # 与训练时的参数一致
# num_classes = 10  # 与训练时的参数一致
# nc = 3  # 与训练时的参数一致
# ngf = 64  # 与训练时的参数一致

# # 实例化 Generator 模型
# netG = _netG(ngpu, nz)  # 或者 _netG_CIFAR10(ngpu, nz)，根据训练时使用的模型选择

# # 载入预训练权重文件到 Generator 模型中
# netG.load_state_dict(torch.load('path_to_generator_weights.pth'))

# # 指定生成图像的目标标签
# # 例如，生成一个标签为 3 的图像
# target_label = 3

# # 生成随机噪声向量
# batch_size = 1  # 生成一张图像
# noise = torch.randn(batch_size, nz, 1, 1)

# # 生成目标标签的 one-hot 编码向量
# target_onehot = torch.zeros(batch_size, num_classes, 1, 1)
# target_onehot[:, target_label, :, :] = 1

# # 将目标标签的 one-hot 编码向量与随机噪声向量拼接起来
# noise_with_label = torch.cat((noise, target_onehot), dim=1)

# # 使用 Generator 生成图像
# with torch.no_grad():
#     fake = netG(noise_with_label)

# # 保存生成的图像
# vutils.save_image(fake.detach(), 'generated_image.png', normalize=True)

#*******************************************************************

ngpu = 1  # 或者根据训练时使用的 GPU 数量来设置
nz = 110  # 与训练时的参数一致
num_classes = 10  # 与训练时的参数一致
nc = 3  # 与训练时的参数一致
ngf = 64  # 与训练时的参数一致
netG = _netG(ngpu, nz)
batch_size = 16

netG.load_state_dict(torch.load('./result/netG_epoch_4950.pth'))

eval_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
eval_noise = Variable(eval_noise)
eval_noise_ = np.random.normal(0, 1, (batch_size, nz))
eval_label = 1
eval_onehot = np.zeros((batch_size, num_classes))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :num_classes] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, nz, 1, 1))

with torch.no_grad():
    fake = netG(eval_noise)

vutils.save_image(fake.detach(), './result/generated_image.png')

