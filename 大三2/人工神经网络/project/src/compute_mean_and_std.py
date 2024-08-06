import torch
from torchvision import datasets, transforms

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为256x256
    transforms.ToTensor()  # 将图像转换为张量，并将像素值归一化到[0, 1]
])

# 加载训练集
train_dataset = datasets.ImageFolder('datasets/Stanford_Dogs/Images', transform=transform)
# train_dataset = datasets.ImageFolder('datasets/CUB_200_2011/CUB_200_2011/images', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 计算均值和标准差
mean = 0.0
std = 0.0
nb_samples = 0

for data in train_loader:
    batch_samples = data[0].size(0)  # 获取批量大小
    data = data[0].view(batch_samples, data[0].size(1), -1)  # 重塑张量
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')
