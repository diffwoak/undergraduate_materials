import numpy as np
from sklearn import svm
import time

train= np.loadtxt(open('mnist_01_train.csv','rb'),delimiter=',',skiprows=1)
test= np.loadtxt(open('mnist_01_test.csv','rb'),delimiter=',',skiprows=1)
# 打乱数据
np.random.shuffle(train)
np.random.shuffle(test)    
train_label = train[:,0]
train_data = train[:,1:]
test_label = train[:,0]
test_data = train[:,1:]

# 线性核SVM初始化与训练
LinearSvc = svm.SVC(C=1.0, kernel='linear')
time_start = time.time()
model1 = LinearSvc.fit(train_data, train_label)
time_end = time.time()
print("time:\t%f"%(time_end-time_start))
print("train:\t%f"%(model1.score(train_data, train_label)))
print("test:\t%f"%(model1.score(test_data, test_label)))
# 高斯核SVM初始化与训练
RbfSvc = svm.SVC(C=1.0, kernel='rbf',gamma='scale')
time_start = time.time()
model2 = RbfSvc.fit(train_data, train_label)
time_end = time.time()
print("time:\t%f"%(time_end-time_start))
print("train:\t%f"%(model2.score(train_data, train_label)))
print("test:\t%f"%(model2.score(test_data, test_label)))