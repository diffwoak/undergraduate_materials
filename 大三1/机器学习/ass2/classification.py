import numpy as np
# import math
import time
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3*32*32, 10)
    def forward(self, x):
        x = x.reshape(-1,3072)
        x = self.l1(x)
        return x
    
class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3*32*32, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256,128)
        self.l4 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.reshape(-1,3072)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l5(x))
        x = self.l4(x)
        return x
    
class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 6, 5) #3*32*32->6*28*28
        self.c2 = nn.Conv2d(6, 16, 5)#6*28*28->16*24*24
        # self.c3 = nn.Conv2d(16,32,3)#16*10*10->32*8*8
        self.l1 = nn.Linear(400, 120)#32*4*4 32*5*5
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 10)
    def forward(self, x):
        x = x.reshape(-1,3,32,32)
        x = F.relu(self.c1(x)) 
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.c2(x))
        # x = F.relu(self.c3(x)) 
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1,400)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

def load_data(dir):
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_train.append(dict[b'data'])
        Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)
    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X_test = dict[b'data']
    Y_test = dict[b'labels']
    X_train = torch.tensor(X_train).float()
    Y_train = torch.tensor(Y_train)
    X_test  = torch.tensor(X_test).float()
    Y_test  = torch.tensor(Y_test)
    return X_train, Y_train, X_test, Y_test

def test(X, Y, net):
    num_data = X.shape[0]
    num_correct = 0
    for i in range(num_data):
        input = X[i]
        output = net(input)
        label = output.max(1, keepdim=True)[1].item()
        true_label = Y[i].item()
        if label == true_label:
            num_correct += 1
    return num_correct / num_data

def train(net,optims,X_train,Y_train,X_test,Y_test):
    num = len(Y_train)
    num_label = 10
    batchsize = 512
    iterations = int(num/batchsize)
    etimes = 150
    epochslist,acclist,losslist=[],[],[]
    startTime = time.time()
    for i in range(iterations*etimes):
        idx = torch.randint(0,num,[batchsize])
        input_b = X_train[idx]
        output_b = net(input_b)
        Y_label = F.one_hot(Y_train[idx],num_label).float()
        loss = nn.CrossEntropyLoss()(output_b,Y_label)
        optims.zero_grad()
        loss.backward()
        optims.step()
        if (i+1) % iterations==0:
            acc = test(X_test,Y_test,net)
            # acc = test(X_train,Y_train,net)
            print('epochs: %d, loss: %.2f, accuracy: %.2f' % ((i+1)/iterations, loss.item(), acc))
            epochslist.append((i+1)/iterations)
            acclist.append(acc)
            losslist.append(loss.item())
    acc = test(X_train,Y_train,net)
    print('--end train accuracy: %.2f' % (acc))
    endTime = time.time()
    print('Total time:{:.4f}'.format(endTime-startTime))
    return epochslist,acclist,losslist

def draw(x,y,labelx,labely):
    plt.figure()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.plot(x, y, color='blue')
    plt.show()

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = load_data(dir='./material')
    # select_net = input("select 0--Linear 1--MLP 2--CNN:")
    select_net = 2
    if select_net==0:net = LinearNet()
    elif select_net==1:net = MLPNet()
    elif select_net==2:net = CNNNet()
    # select_opt = input("select 0--SGD 1--SGD momentum 2--Adam:")
    select_opt = 2
    if select_opt==0:optims = optim.SGD(net.parameters(), lr=0.001,momentum=0)
    elif select_opt==1:optims = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)
    elif select_opt==2:optims = optim.Adam(net.parameters())
    epochslist,acclist,losslist = train(net,optims,X_train,Y_train,X_test,Y_test)
    draw(epochslist,acclist,'epochs','acc')
    draw(epochslist,losslist,'epochs','loss')