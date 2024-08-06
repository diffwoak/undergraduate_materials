import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

from trainer import VolumeClassifier
from config import INIT_TRAINER
from config import VERSION
from data_utils.csv_reader import csv_reader_single
import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

epsilons = [0, .05, .1, .15, .2, .25, .3]
# epsilons = [0, .05]

INIT_TRAINER['pre_trained'] = True
INIT_TRAINER['batch_size'] = 10
classifier = VolumeClassifier(**INIT_TRAINER)

# Set random seed for reproducibility
torch.manual_seed(0)

test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
                # test_csv_path = './csv_file/Stanford_Dogs_test.csv'
label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')   # label_dict = path : label
test_path = list(label_dict.keys())     # 测试集路径path

test_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.CenterCrop(cropped=classifier.image_size),
            tr.ToTensor(),
            tr.Normalize(classifier.train_mean, classifier.train_std)
            ])

test_dataset = DataGenerator(test_path,
                            label_dict,
                            transform=test_transformer)

test_loader = DataLoader(test_dataset,
                        batch_size=classifier.batch_size,
                        shuffle=False,
                        num_workers=classifier.num_workers,
                        pin_memory=True)

net = classifier.net
model = net.cuda()
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, 
           mean=[0.48560741861744905, 0.49941626449353244, 0.43237713785804116], 
           std=[0.2321024260764962, 0.22770540015765814, 0.2665100547329813]):

    if isinstance(mean, list):
        mean = torch.tensor(mean).cuda()
    if isinstance(std, list):
        std = torch.tensor(std).cuda()

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test(model, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for step, sample in enumerate(test_loader):
        data = sample['image']
        target = sample['label']

        data, target = data.cuda(), target.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        output = F.softmax(output, dim=1)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        correct_indices = init_pred.eq(target.view_as(init_pred)).squeeze()

        print(f'step:{step},test_acc{correct_indices.sum().item()/classifier.batch_size}')

        # Calculate the loss
        loss = nn.CrossEntropyLoss()

        # Zero all existing gradients
        model.zero_grad()

        loss = loss(output, target)

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data, classifier.train_mean, classifier.train_std)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize(classifier.train_mean, classifier.train_std)(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        adv_correct_indices = final_pred.eq(target.view_as(final_pred)).squeeze()
        # adv_incorrect_indices = final_pred.ne(target.view_as(final_pred)).squeeze()
        adv_incorrect_indices = correct_indices & final_pred.ne(target.view_as(final_pred)).squeeze()


        print(f'epsilon:{epsilon},step:{step},adv_attack_acc:{adv_correct_indices.sum().item()/classifier.batch_size}')

        correct += adv_correct_indices.sum().item()

        # 保存一些对抗样本以供后续可视化
        if epsilon==0:
            if len(adv_examples) < 5:
                adv_indices = adv_correct_indices.nonzero().squeeze().tolist()
                if isinstance(adv_indices, int):  # 处理只有一个正确预测的情况
                    adv_indices = [adv_indices]

                for idx in adv_indices[:5 - len(adv_examples)]:
                    adv_ex = perturbed_data[idx].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[idx].item(), final_pred[idx].item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_indices = adv_incorrect_indices.nonzero().squeeze().tolist()
                if isinstance(adv_indices, int):  # 处理只有一个正确预测的情况
                    adv_indices = [adv_indices]

                for idx in adv_indices[:5 - len(adv_examples)]:
                    adv_ex = perturbed_data[idx].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[idx].item(), final_pred[idx].item(), adv_ex))


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

pic_save_dir = 'res_pic'

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(pic_save_dir, f'{classifier.net_name}_Acc_vs_Epsilon.jpg'))

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        if ex.ndim == 3:  # Check if the image has color channels
            plt.imshow(ex.transpose((1, 2, 0)))  # Assuming `ex` is a tensor
        else:
            plt.imshow(ex, cmap='gray')  # For grayscale images   
    
plt.tight_layout()
plt.savefig(os.path.join(pic_save_dir, f'{classifier.net_name}_adv_attack_example.jpg'))