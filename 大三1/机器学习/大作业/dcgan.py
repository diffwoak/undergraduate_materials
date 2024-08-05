import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

def draw_loss(G_loss,D_loss,epoch):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(epoch+1), G_loss, label="generator loss")
    plt.plot(range(epoch+1), D_loss, label="discriminator loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_{}.png".format(epoch), dpi=400)
    # plt.show()
def draw_D(D_x,D_Gz,epoch):
    plt.figure()
    plt.xlabel("Epochs")
    plt.plot(range(epoch+1), D_x, label="D(x)")
    plt.plot(range(epoch+1), D_Gz, label="D(G(z))")
    plt.legend()
    plt.grid()
    plt.savefig("D_{}.png".format(epoch), dpi=400)
    # plt.show()

if __name__=='__main__':
    # 初始化
    latent_dim = 1000
    img_channels = 3
    img_size = 28
    batch_size = 32
    epochs = 100
    lr = 0.0002

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # 使用训练过的生成器模型实例
    # generator = Generator(latent_dim, img_channels) 
    # generator.load_state_dict(torch.load("dcgan_generator.pth"))
    # generator.eval()
    # gen_imgs = generator(torch.randn(24, latent_dim, 1, 1, device=device))
    # save_image(gen_imgs, f"gen_test.png", normalize=True)
    generator = Generator(latent_dim, img_channels).to(device)  # 生成器
    discriminator = Discriminator(img_channels).to(device)      # 判别器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)) # betas衰减率
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()        # 二分类交叉熵损失
    # 数据集
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    G_loss,D_loss=[],[]
    D_x,D_Gz = [],[]
    # 训练模型
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            if i == len(dataloader)-1: continue
            real_imgs, _ = data
            real_imgs = real_imgs.to(device)
            # 训练判别器
            optimizer_D.zero_grad()
            real_labels = torch.full((batch_size, 1), 1.0, device=device)
            fake_labels = torch.full((batch_size, 1), 0.0, device=device)
            # 真实图像的损失
            output_real = discriminator(real_imgs).resize(batch_size,1)
            loss_real = criterion(output_real, real_labels)
            # 假图像损失
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach()).resize(batch_size,1)
            loss_fake = criterion(output_fake, fake_labels)
            # 总损失
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            D_x.append(output_real.mean().item())
            # 训练生成器
            optimizer_G.zero_grad()
            output = discriminator(fake_imgs).resize(batch_size,1)
            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizer_G.step()
            D_Gz.append(output.mean().item())

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}]  [Batch {i}/{len(dataloader)}]    \t[D loss: {loss_D.item():.4f}  G loss: {loss_G.item():.4f}]")
        G_loss.append(loss_G.item()),D_loss.append(loss_D.item())
        # 保存生成的图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_imgs = generator(torch.randn(24, latent_dim, 1, 1, device=device))
                save_image(fake_imgs, f"generated_epoch_{epoch}.png", normalize=True)
                draw_loss(G_loss,D_loss,epoch)

    draw_loss(G_loss,D_loss,epochs-1)
    # 保存训练好的生成器模型
    torch.save(generator.state_dict(), "dcgan_generator.pth")