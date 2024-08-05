import matplotlib.pyplot as plt
import time
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def acc(func,x,y,w):
    p=np.dot(x,w)
    if func=='hinge':
        for i in range(x.shape[0]):
            if p[i]>=0:p[i]=1
            else:p[i]=-1
    elif func=='cross':
        for i in range(x.shape[0]):
            if p[i]>=0:p[i] = 1
            else:p[i] = 0
    count=0
    for i in range(y.shape[0]):
        if y[i]==p[i]:count+=1
    return count/y.shape[0]    

def descent(func, w, train_data, train_label,learning_rate):
    if func=='hinge':
        a = np.dot(train_data,w)
        dw=np.zeros(w.shape[0])
        for j in range(a.shape[0]):
            a[j]=a[j]*train_label[j] #wx*y
            if a[j]<1:
               dw-=train_data[j]*train_label[j] # -x*y
        dw/=train_data.shape[0] # -x*y/N
        dw = np.reshape(dw, (-1,1))
        w-=learning_rate*dw # w-= -x*y/N
        # 损失函数 loss = max(0,1-wx*y)
        loss = np.zeros(train_data.shape[0])
        for j in range(loss.shape[0]):
            if a[j]<1:
                loss[j]=1-a[j]
        loss=sum(loss)/train_data.shape[0]
        return w,loss
    elif func=='cross':
        a = np.dot(train_data, w)
        # a = multi(train_data,w)
        dw=np.zeros(w.shape[0])
        loss = 0.0
        for i in range(a.shape[0]):
            a[i,0] = sigmoid(a[i,0])
            if int(train_label[i]) == 1:
                loss += np.log(a[i,0])
            else:
                if a[i,0]<1:
                    loss += np.log(1-a[i,0])
            dw -= train_data[i]*(train_label[i]-a[i,0]) 
        dw /= train_data.shape[0]
        dw = np.reshape(dw, (-1,1))
        w -= learning_rate*dw
        loss *= -(1/train_data.shape[0])
        return w,loss
    
def draw(x,lab):
    plt.figure(lab)
    plt.xlabel("epochs")
    plt.ylabel(lab)
    plt.plot(range(epochs), x)
    plt.show()   

if __name__ == "__main__":
    # 超参数设置
    epochs = 100
    learning_rate = 0.05
    mode = input("select 1--hinge loss  2--cross-entropy loss: ")
    func = ''
    if mode == '1':func = 'hinge'
    elif mode == '2':func = 'cross'
    train= np.loadtxt(open('mnist_01_train.csv','rb'),delimiter=',',skiprows=1)
    test= np.loadtxt(open('mnist_01_test.csv','rb'),delimiter=',',skiprows=1)
    np.random.shuffle(train)
    np.random.shuffle(test)
    train_label=np.empty(train.shape[0])
    train_data=np.empty(shape=[train.shape[0],train.shape[1]])
    test_label=np.empty(test.shape[0])
    test_data=np.empty(shape=[test.shape[0],test.shape[1]]); 
    # 标签 hinge{-1，1} cross{0,1}
    for i in range(train.shape[0]):
        train_label[i]=train[i][0]
        if train_label[i]==0 and func=='hinge':
            train_label[i]=-1
        train_data[i]=train[i][:]
        train_data[i][0]=1
    for i in range(test.shape[0]):
        test_label[i]=test[i][0]
        if test_label[i]==0 and func=='hinge':
            test_label[i]=-1
        test_data[i]=test[i][:]
        test_data[i][0]=1
    # 特征标准化
    m,s = [],[]
    for i in range(train_data.shape[1]):
        m.append(np.mean(train_data[:, i]))
        s.append(np.std(train_data[:,i]))
        if s[i] != 0:
            train_data[:, i] = (train_data[:,i]-m[i])/s[i]
            test_data[:, i] = (test_data[:,i]-m[i])/s[i] #测试集使用训练集的平均值和标准差 
    # 设置参数矩阵
    np.random.seed(0)
    W = np.random.rand(train_data.shape[1],1)
    loss_show=[]
    acc_show=[]
    start=time.time()
    for i in range(epochs):
        W,loss = descent(func, W, train_data, train_label,learning_rate)
        loss_show.append(loss)
        acc_test = acc(func,test_data,test_label,W)
        acc_show.append(acc_test)
        if i%10==0:
            # print(i,loss,acc(func,train_data,train_label,W))
            print("epochs:%d\tloss:%f\tacc:%f"%(i,loss,acc(func,train_data,train_label,W)))
    end=time.time()
    print('test acc:%f\ttrain time:%f s'%(acc(func,test_data,test_label,W),end-start))
    draw(loss_show,"loss")
    draw(acc_show,"accuary")