import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fsize=3
        self.sec=True   #是否使用第二个卷积层
        self.lsize=8*(28-self.fsize+1)*(28-self.fsize+1)
        if self.sec:
            self.lsize=16*(28-2*(self.fsize-1))*(28-2*(self.fsize-1))
        self.conv_layer1 = nn.Conv2d(1,8,(self.fsize,self.fsize))
        self.conv_layer2 = nn.Conv2d(8,16,(self.fsize,self.fsize))
        #self.pool_layer = nn.MaxPool2d(2, stride=1)
        self.linear_layer1 = nn.Linear(self.lsize, 100)
        self.linear_layer2 = nn.Linear(100,10)
    def forward(self, x):
        x = torch.relu(self.conv_layer1(x))
        #x = self.pool_layer(x) #池化
        if self.sec:    #使用第二个卷积层
            x = torch.relu(self.conv_layer2(x)) 
        #print(x.shape) #查看
        x = x.reshape(-1,self.lsize)
        x = torch.relu(self.linear_layer1(x))
        x = self.linear_layer2(x)
        return torch.softmax(x,dim=-1)


def test(data, labels, net):
    num_data = data.shape[0]
    num_correct = 0
    for i in range(num_data):
        feature = data[i]
        prob = net(feature).detach()
        dist = Categorical(prob)
        label = dist.sample().item()
        true_label = labels[i].item()
        if label == true_label:
            num_correct += 1
    return num_correct / num_data

def draw(x,y):
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.plot(x, y, color='blue')
    plt.show()


if __name__=="__main__":
    is_train = False
    train_data = torch.load('train_data.pth')
    train_labels = torch.load('train_labels.pth')
    test_data = torch.load('test_data.pth')
    test_labels = torch.load('test_labels.pth')

    num_data = train_data.shape[0]
    num_labels = 10
    batchsize = 100
    iterations = int(num_data/batchsize)
    etimes = 20
    milestones = [6*iterations, 8*iterations]
    epochs=[] #画图
    ac=[]

    net = MyConvNet()
    if is_train:
        sgd_optim = optim.SGD(net.parameters(), lr=0.02)
        scheduler_lr = optim.lr_scheduler.MultiStepLR(sgd_optim, milestones=milestones, gamma=0.5)
        t=scheduler_lr.get_last_lr()
        #print(t) 
        for i in range(iterations*etimes):
            idx = torch.randint(0, num_data, [batchsize])
            feature = train_data[idx]
            prob = net(feature)
            true_label = F.one_hot(train_labels[idx], num_labels).float()
            loss = nn.CrossEntropyLoss()(prob, true_label)
            sgd_optim.zero_grad()        
            loss.backward()         
            sgd_optim.step()
            scheduler_lr.step()
            #print('t= %.4f ,i= %d ' % (t[0],i))   
            #t1=scheduler_lr.get_last_lr()
            if t!=scheduler_lr.get_last_lr():
                t=scheduler_lr.get_last_lr()
                print('lr=%.4f' % (t[0]))
            if i % iterations ==0:
                acc = test(test_data, test_labels, net)
                print('epochs: %d, loss: %.2f, accuracy: %.2f' % (i/iterations, loss.item(), acc))
                epochs.append(i/iterations)
                ac.append(acc)
        draw(epochs,ac)
        torch.save(net.state_dict(), 'hw7_21307347_chenxinyu.pth')
    else:
        net.load_state_dict(torch.load('hw7_21307347_chenxinyu.pth'))
        acc = test(test_data, test_labels, net)
        acc1 = test(train_data, train_labels, net)
        print('accuracy: %.4f' % (acc))
        print('accuracy_train: %.4f' % (acc1))