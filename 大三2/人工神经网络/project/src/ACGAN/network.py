import torch
import torch.nn as nn
import torch.nn.functional as F


# class _netG(nn.Module):
#     def __init__(self, ngpu, nz):
#         super(_netG, self).__init__()
#         self.ngpu = ngpu
#         self.nz = nz

#         # first linear layer
#         self.fc1 = nn.Linear(110, 768)
#         # Transposed Convolution 2
#         self.tconv2 = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
#             nn.BatchNorm2d(384),
#             nn.ReLU(True),
#         )
#         # Transposed Convolution 3
#         self.tconv3 = nn.Sequential(
#             nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#         )
#         # Transposed Convolution 4
#         self.tconv4 = nn.Sequential(
#             nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
#             nn.BatchNorm2d(192),
#             nn.ReLU(True),
#         )
#         # Transposed Convolution 5
#         self.tconv5 = nn.Sequential(
#             nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#         )
#         # Transposed Convolution 5
#         self.tconv6 = nn.Sequential(
#             nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
#             nn.Tanh(),
#         )

#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             input = input.view(-1, self.nz)
#             fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
#             fc1 = fc1.view(-1, 768, 1, 1)
#             tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
#             tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
#             tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
#             tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
#             tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
#             output = tconv5
#         else:
#             input = input.view(-1, self.nz)
#             fc1 = self.fc1(input)
#             fc1 = fc1.view(-1, 768, 1, 1)
#             tconv2 = self.tconv2(fc1)
#             tconv3 = self.tconv3(tconv2)
#             tconv4 = self.tconv4(tconv3)
#             tconv5 = self.tconv5(tconv4)
#             tconv5 = self.tconv6(tconv5)
#             output = tconv5
#         return output


# class _netD(nn.Module):
#     def __init__(self, ngpu, num_classes=10):
#         super(_netD, self).__init__()
#         self.ngpu = ngpu

#         # Convolution 1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 16, 3, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # Convolution 2
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # Convolution 3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # Convolution 4
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # Convolution 5
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # Convolution 6
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5, inplace=False),
#         )
#         # discriminator fc
#         self.fc_dis = nn.Linear(13*13*512, 1)
#         # aux-classifier fc
#         self.fc_aux = nn.Linear(13*13*512, num_classes)
#         # softmax and sigmoid
#         self.softmax = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
#             conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
#             conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
#             conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
#             conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
#             conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
#             flat6 = conv6.view(-1, 13*13*512)
#             fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
#             fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
#         else:
#             conv1 = self.conv1(input)
#             conv2 = self.conv2(conv1)
#             conv3 = self.conv3(conv2)
#             conv4 = self.conv4(conv3)
#             conv5 = self.conv5(conv4)
#             conv6 = self.conv6(conv5)

#             flat6 = conv6.view(-1, 13*13*512)
#             fc_dis = self.fc_dis(flat6)
#             fc_aux = self.fc_aux(flat6)
#         classes = F.softmax(fc_aux, dim=1)
#         realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
#         return realfake, classes

class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # 第一个全连接层
        self.fc1 = nn.Linear(nz, 256 * 4 * 4)
        # 第一个转置卷积层
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 输出: (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # 第二个转置卷积层
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 输出: (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # 第三个转置卷积层
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),    # 输出: (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # 第四个转置卷积层
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),     # 输出: (3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 256, 4, 4)
        tconv1 = self.tconv1(fc1)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        output = self.tconv4(tconv3)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD, self).__init__()
        self.ngpu = ngpu

        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),   # 输出: (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 输出: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 输出: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 第四个卷积层
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 输出: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 判别器全连接层
        self.fc_dis = nn.Linear(4*4*512, 1)
        # 辅助分类器全连接层
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax 和 sigmoid
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        flat4 = conv4.view(-1, 4*4*512)
        fc_dis = self.fc_dis(flat4)
        fc_aux = self.fc_aux(flat4)
        
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes



class _netG_CIFAR10(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            output = tconv5
        return output


class _netD_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
        classes = F.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes
