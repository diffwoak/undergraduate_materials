import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

import model.gan as gan
from config import CUB_TRAIN_MEAN, CUB_TRAIN_STD
from data_utils.csv_reader import csv_reader_single

from config_gan import PRE_TRAINED, GEN_PATH, DIS_PATH, DATA_PATH, INIT_TRAINER,DATA_NUM,GEN_PIC,DEVICE

device = torch.device('cuda:{}'.format(DEVICE) if torch.cuda.is_available() else 'cpu')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def gan_trainer(image_size, encoding_dims, batch_size, epochs, num_workers, gen_path = None, dis_path = None):

    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    generator = gan.DCGANGenerator(encoding_dims=encoding_dims, out_size=image_size, out_channels=3).cuda(DEVICE)
    discriminator = gan.DCGANDiscriminator(in_size=image_size, in_channels=3).cuda(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    # add pth
    if PRE_TRAINED and os.path.exists(gen_path):
        checkpoint = torch.load(gen_path)
        generator.load_state_dict(checkpoint)
    if PRE_TRAINED and os.path.exists(gen_path):
        checkpoint = torch.load(dis_path)
        generator.load_state_dict(checkpoint)
    

    csv_path = './csv_file/cub_200_2011.csv_train.csv'
    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
    train_path = list(label_dict.keys())
    
    train_transformer = transforms.Compose([
        tr.ToCVImage(),
        tr.RandomResizedCrop(image_size),
        tr.ToTensor(),
        tr.Normalize(CUB_TRAIN_MEAN, CUB_TRAIN_STD)
    ])
    
    train_dataset = DataGenerator(train_path,
                                  label_dict,
                                  transform=train_transformer)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    
    generator = generator.cuda(DEVICE)
    discriminator = discriminator.cuda(DEVICE)
    
    optimG = torch.optim.AdamW(generator.parameters(), 0.0001, betas = (0.5,0.999))
    optimD = torch.optim.AdamW(discriminator.parameters(), 0.0002, betas = (0.5,0.999))
    
    loss = nn.BCELoss()
    
    for epoch in range(1,epochs+1):
        for step, sample in enumerate(train_loader, 0):
            
            images = sample['image'].to(device)
            bs= images.size(0)

            # ---------------------
            #         disc
            # ---------------------
            optimD.zero_grad()       

            # real

            pvalidity = discriminator(images)
            pvalidity = F.sigmoid(pvalidity)
            errD_real = loss(pvalidity, torch.full((bs,), 1.0, device=device))         
            errD_real.backward()

            # fake 
            noise = torch.randn(bs, encoding_dims, device=device)  
            fakes = generator(noise)
            pvalidity = discriminator(fakes.detach())
            pvalidity = F.sigmoid(pvalidity)

            errD_fake = loss(pvalidity, torch.full((bs,), 0.0, device=device))
            errD_fake.backward()

            # finally update the params
            errD = errD_real + errD_fake

            optimD.step()
        

            # ------------------------
            #      gen
            # ------------------------
            optimG.zero_grad()

            noise = torch.randn(bs, encoding_dims, device = device)   
            fakes = generator(noise)
            pvalidity = discriminator(fakes)
            pvalidity = F.sigmoid(pvalidity)

            errG = loss(pvalidity, torch.full((bs,), 1.0, device=device))        
            errG.backward()

            optimG.step()
        
            print("[{}/{}] [{}/{}] G_loss: [{:.4f}] D_loss: [{:.4f}]"
              .format(epoch, epochs, step, len(train_loader), errG, errD))
            
        torch.save(generator.state_dict(),'ckpt/{}_generator.pth'.format(epoch))
#         torch.save(discriminator.state_dict(),'ckpt/{}_discriminator.pth'.format(epoch))
    torch.save(generator.state_dict(),gen_path)
    torch.save(discriminator.state_dict(),dis_path)

if __name__ == "__main__":
    
    
    if GEN_PIC and os.path.exists(GEN_PATH):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        generator = gan.DCGANGenerator(encoding_dims=INIT_TRAINER['encoding_dims'], out_size=INIT_TRAINER['image_size'], out_channels=3)
        generator = generator.cuda(DEVICE)
        checkpoint = torch.load(GEN_PATH)
        generator.load_state_dict(checkpoint)
        
        noise = torch.randn(DATA_NUM,INIT_TRAINER['encoding_dims'],device = device)  
        gen_images = generator(noise).detach()
        
        for i in range(DATA_NUM):
            save_image(gen_images[i], DATA_PATH+str(i)+'.jpg')

    else:
        gan_trainer(**INIT_TRAINER)