import numpy as np
import pandas as pd
import random
import time
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.linalg import pinvh
from munkres import Munkres, print_matrix
import matplotlib.pyplot as plt
EPOCHS = 250
NEW_DIMENSION = 60

class Kmeans():
    def __init__(self, data, K, cen_init):
        self.K = K
        self.dimension = data.shape[1]      #dimension = 60
        self.centroids = None               #聚类中心
        self.clusters = None                #训练集分类情况
        self.initCentroids(data, cen_init)  #初始化聚类中心'random', 'distance'
    def initCentroids(self, data, cen_init):    #初始化聚类中心
        indexes = np.arange(data.shape[0])
        np.random.shuffle(indexes)
        self.centroids = np.zeros((self.K, data.shape[1])) # K x 60
        if cen_init=='random':
            for i in range(self.K):
                self.centroids[i] = data[indexes[i]]
        elif cen_init=='distance':
            usedIndexes = list()
            for i in range(self.K):
                if i==0:
                    self.centroids[i] = data[indexes[i]]
                    usedIndexes.append(indexes[i])
                else:
                    distances = self.getDistance(data)  #与聚类中心距离 60000 x K
                    totalDistances = np.sum(distances, axis=1) #距离和 60000 x 1
                    indexes = np.argsort(-totalDistances)      #排序 由大到小
                    for index in indexes:
                        if index not in usedIndexes:
                            self.centroids[i] = data[index]
                            usedIndexes.append(index)
                            break
        self.clusters,_ = self.getClusters(data)
    def getDistance(self, data):    #计算所有点到当前聚类中心的距离
        distances = np.zeros((data.shape[0], self.centroids.shape[0]))  #60000 x k
        for i in range(data.shape[0]):
            distances[i] = np.sum((self.centroids - data[i])**2, axis=1)**0.5
        return distances
    def getClusters(self, data):    #划分样本点，计算loss
        distances = self.getDistance(data)      # 60000 x k
        clusters = np.argmin(distances, axis=1) # 1 x 60000
        avgDistances = np.sum(np.min(distances, axis=1))/data.shape[0]  #所有样本点与聚类中心的平均距离loss
        return clusters, avgDistances
    def getCentroids(self, data, clusters): #更新聚类中心
        oneHotClusters = np.zeros((data.shape[0], self.K))  #60000 x k
        for i in range(data.shape[0]):
            oneHotClusters[i,clusters[i]] = 1 #clusters：60000 x 1
        CluSumDis = np.dot(oneHotClusters.T, data)  # k x 60000 * 60000 x 60 = k x 60 每类距离和
        CluNum = np.sum(oneHotClusters, axis=0).reshape((-1,1)) # k x 1 每类数量
        return CluSumDis/CluNum #  k x 60 / k x 1 对应行相乘
    def getAccuracy(self, clusters, Labels):  #计算结果准确率
        clustersType = np.unique(clusters)  #1 x k
        LabelType = np.unique(Labels)       #1 x k
        labelNum = np.maximum(len(clustersType), len(LabelType))
        costMatrix = np.zeros((labelNum, labelNum)) # 代价矩阵
        for i in range(len(clustersType)):
            selclusters = (clusters==clustersType[i]).astype(float) #逻辑转浮点 1 x 60000
            for j in range(len(LabelType)):
                sellabels = (Labels==LabelType[j]).astype(float)    # 1 x 60000
                costMatrix[i,j] = -np.sum(selclusters*sellabels)    # 越小匹配度越高
        m = Munkres()
        indexes = m.compute(costMatrix) # 匈牙利算法->索引映射
        maplabels = np.zeros_like(clusters, dtype=int)
        for index1,index2 in indexes:
            if index1<len(clustersType) and index2<len(LabelType):
                maplabels[clusters==clustersType[index1]] = LabelType[index2] # 映射结束
        return np.sum((maplabels==Labels).astype(float))/Labels.size
    def train(self,data):   #训练函数
        newcentroids = self.getCentroids(data, self.clusters)
        diff = np.sum((newcentroids-self.centroids)**2)**0.5
        self.centroids = newcentroids
        self.clusters, loss = self.getClusters(data)
        return loss,diff
    def test(self,data,labels): #测试函数
        clusters, loss = self.getClusters(data)
        acc = self.getAccuracy(clusters, labels)
        return acc, loss

class GMM():
    def __init__(self, K, data, init_type='random', cov_type='commom', reg_covar=1e-6):
        self.K = K
        self.dimension = data.shape[1]
        self.means = None
        self.weights = None
        self.cov = None
        self.cov_type = cov_type
        self.reg_covar = reg_covar
        self._init_parameters(init_type=init_type, cov_type=cov_type, data=data)
    def _init_parameters(self, data, init_type='random', cov_type='full'):  #初始化均值、协方差、权重
        # init_type:'random', 'randgamma'
        # cov_type:'commom', 'circle', 'ellipse'
        if init_type=='random':
            indexes = np.arange(data.shape[0])
            np.random.shuffle(indexes)
            self.means = np.zeros((self.K, self.dimension))  # k x 50
            self.means = data[indexes[:self.K]]
            self.weights = np.ones(self.K)/self.K
            tempCov = np.cov(data, rowvar=False)
            tempCov += np.eye(self.dimension)*self.reg_covar
            if cov_type=='commom':      #普通矩阵
                self.cov = tempCov[np.newaxis,:].repeat(self.K, axis=0) # k x 50 x 50
            elif cov_type=='circle':    #对角且元素值都相等
                self.cov = np.ones(self.K)*np.diag(tempCov).mean()  #取样本得到矩阵平均值
            elif cov_type=='ellipse':   #对角但元素值不要求相等
                self.cov = np.diag(tempCov)
                self.cov = self.cov[np.newaxis, :].repeat(self.K, axis=0)                
        elif init_type=='randgamma':
            gamma = np.random.rand(data.shape[0], self.K)   # 60000 x k
            gamma /= np.sum(gamma, axis=1).reshape(-1,1)    #每行gamma和为1
            self.MStep(data, gamma)
    def EStep(self, data):
        gamma = np.zeros((data.shape[0], self.K))
        Cov = self.getfullcov()
        for k in range(self.K):
            gamma[:,k] = self.weights[k]*self.gaussfunc(data, self.means[k], Cov[k])
        gamma /= np.sum(gamma, axis=1).reshape(-1,1)
        return gamma
    def MStep(self, data, gamma):
        self.means = np.dot(gamma.T, data)/np.sum(gamma, axis=0).reshape(-1,1)
        self.weights = np.sum(gamma, axis=0)/data.shape[0]
        if self.cov_type=='commom': #k x 50 x 50
            self.cov = np.zeros((self.K, self.dimension, self.dimension))
            for k in range(self.K):
                diff = data - self.means[k]
                self.cov[k] = np.dot(gamma[:,k]*diff.T, diff)
                self.cov[k] /= np.sum(gamma[:,k])
                self.cov[k] += np.eye(self.dimension)*self.reg_covar # 对角非负正则化
        elif self.cov_type=='circle':   # k x 1
            self.cov = np.zeros((self.K))
            for k in range(self.K):
                diff = data - self.means[k]
                temp = np.dot(gamma[:,k]*diff.T, diff)
                temp /= np.sum(gamma[:,k])
                temp += np.eye(self.dimension)*self.reg_covar # 对角非负正则化
                self.cov[k] = np.diag(temp).mean()
        elif self.cov_type=='ellipse':
            self.cov = np.zeros((self.K, self.dimension))   # k x 50
            for k in range(self.K):
                diff = data - self.means[k]
                temp = np.dot(gamma[:,k]*diff.T, diff)
                temp /= np.sum(gamma[:,k])
                temp += np.eye(self.dimension)*self.reg_covar # 对角非负正则化
                self.cov[k] = np.diag(temp)
    def gaussfunc(self, x, mean, cov):  #计算高斯函数
        diff = x - mean
        expon = -0.5*(np.sum(np.dot(diff,np.linalg.pinv(cov))*diff, axis=1))
        return np.exp(expon)/(((2*np.pi)**(self.dimension/2))*(np.sqrt(np.linalg.det(cov))))
    def getAccuracy(self, clusters, Labels): #计算结果准确率
        clusterType = np.unique(clusters)     # 1 x k
        LabelType = np.unique(Labels)   # 1 x k
        labelNum = np.maximum(len(clusterType), len(LabelType))
        costMatrix = np.zeros((labelNum, labelNum))# 代价矩阵
        for i in range(len(clusterType)):
            selclusters = (clusters==clusterType[i]).astype(float) #逻辑转浮点 1 x 60000
            for j in range(len(LabelType)):
                sellabels = (Labels==LabelType[j]).astype(float)    # 1 x 60000
                costMatrix[i,j] = -np.sum(selclusters*sellabels)    # 越小匹配度越高
        m = Munkres()
        indexes = m.compute(costMatrix)
        maplabels = np.zeros_like(clusters, dtype=int)
        for index1,index2 in indexes:
            if index1<len(clusterType) and index2<len(LabelType):
                maplabels[clusters==clusterType[index1]] = LabelType[index2]
        return np.sum((maplabels==Labels).astype(float))/Labels.size
    def train(self, data,gam):  #训练函数
        gamma = self.EStep(data)
        diff = np.linalg.norm(gamma-gam)
        self.MStep(data, gamma)
        return gamma,diff
    def test(self, data, labels):   #测试函数
        gamma = self.EStep(data)
        clusters = np.argmax(gamma, axis=1)
        return self.getAccuracy(clusters, labels)
    def getfullcov(self):   #计算完整协方差矩阵
        tempCov = None  
        if self.cov_type=='commom':
            tempCov = self.cov
        elif self.cov_type=='circle':
            tempCov = np.zeros((self.K, self.dimension, self.dimension))
            for k in range(self.K):
                tempCov[k] = np.eye(self.dimension)*self.cov[k]
        elif self.cov_type=='ellipse':
            tempCov = np.zeros((self.K, self.dimension, self.dimension))
            for k in range(self.K):
                tempCov[k] = np.diag(self.cov[k])
        return tempCov

if __name__=='__main__':
    train_data = pd.read_csv("data/mnist_train.csv")
    test_data = pd.read_csv("data/mnist_test.csv")
    TrainData = train_data.iloc[:,1:].values
    train_labels = train_data.iloc[:,0].values
    TestData = test_data.iloc[:,1:].values
    test_labels = test_data.iloc[:,0].values
    # 数据降维
    pcaModel = PCA(n_components=NEW_DIMENSION)
    pcaModel.fit(TrainData)
    TrainData = pcaModel.transform(TrainData) #(60000, 60)
    TestData = pcaModel.transform(TestData)   #(10000, 60)
    
    acc_com,acc_cir=[],[]
    temepoch = 0

    np.random.seed(1)
    k = Kmeans(TrainData,10,'distance')
    start_time = time.time()
    for i in range(EPOCHS):
        loss,diff = k.train(TrainData)
        # acc = k.getAccuracy(k.clusters, train_labels)
        acc, loss = k.test(TestData, test_labels)
        print('epochs:{}\tloss = {:.4f}\tdiff={:.4f}\tacc = {:.4f}'.format(i+1, loss,diff, acc))
        acc_cir.append(acc);temepoch=i+1
        if diff < 1e-7:
            break
    end_time = time.time()
    acc, loss = k.test(TestData, test_labels)
    print('Test: acc = {:.4f}\tepoch = {}\ttime:{:.2f}s'.format(acc,temepoch,end_time-start_time)) 

    # GMM模型
    # init_type:'random', 'randgamma'
    #cov_type:'commom', 'circle', 'ellipse'
    np.random.seed(1)
    gmmModel = GMM(10,TrainData,init_type='random',cov_type='commom')
    start_time = time.time()
    gamma = np.zeros((TrainData.shape[0], 10))  #用于计算是否收敛
    for i in range(EPOCHS):
        gamma,diff = gmmModel.train(TrainData,gamma)
        acc = gmmModel.test(TestData, test_labels)
        acc_com.append(acc);temepoch=i+1
        # print('epochs:{}\tdiff:{:.4f}\tacc = {:.4f}'.format(i+1,diff,acc)) 
        if diff < 1e-3:
            break
    end_time = time.time()
    acc = gmmModel.test(TestData, test_labels)
    print('Test: acc = {:.4f}\tepoch = {}\ttime:{:.2f}s'.format(acc,temepoch,end_time-start_time)) 


    l = max(len(acc_com),len(acc_cir))
    while len(acc_com)<l or len(acc_cir)<l:
        if len(acc_com)<l:
            acc_com.append(acc_com[len(acc_com)-1])
        if len(acc_cir)<l:
            acc_cir.append(acc_cir[len(acc_cir)-1])
        # if len(acc_ell)<l:
        #     acc_ell.append(acc_ell[len(acc_ell)-1])
    x = np.linspace(0,l,l)
    plt.figure()
    plt.plot(x,acc_com)
    plt.plot(x,acc_cir)
    # plt.plot(x,acc_ell)
    plt.legend(['acc_GMM','acc_KMeans'])
    plt.show()