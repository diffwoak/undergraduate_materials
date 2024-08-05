import random
import numpy as np
import matplotlib.pyplot as plt

def draw(distance_real,distance_best):
    best=[]
    for d in distance_best:
        best.append(np.linalg.norm(d - distance_best[-1], ord=2))
    plt.plot(distance_real, label='distance_real')
    plt.plot(best, label='distance_best')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lamb = 0.01 # 正则项参数
    X_true = np.zeros(200)
    index_X = random.sample(range(200), 5)
    for index in index_X:
        X_true[index] = np.random.normal(0, 1)
    E = np.random.normal(0, 0.1, 5*10).reshape(10,5)
    A = np.random.normal(0, 1, 5*200*10).reshape(10,5,200)
    B = np.zeros((10,5))
    for i in range(10):
        B[i] = A[i] @ X_true + E[i]    
# 邻近点梯度法
    distance_real,distance_best = [],[]
    alpha = 0.001
    X_pred = np.zeros(200)
    while 1:
        delta_X = 0 
        loss = 0
        for i in range(10):
            pred_B = A[i] @ X_pred
            delta_X += (pred_B - B[i]) @ A[i]         
        X_half = X_pred - alpha*delta_X
        X_next = np.where(X_half > alpha*lamb, X_half - alpha*lamb, np.where(X_half < alpha*lamb, X_half + alpha*lamb, 0))
        print(np.linalg.norm(X_pred - X_next, ord=2))
        if np.linalg.norm(X_pred - X_next, ord=2) < 10e-5: break
        X_pred = X_next
        distance_real.append(np.linalg.norm(X_next - X_true, ord=2))
        distance_best.append(X_next)
    draw(distance_real,distance_best)
# 交替方向乘子法
    distance_real,distance_best = [],[]
    v = np.zeros((10,200))
    C = 0.05
    X_pred = np.zeros((10,200))
    Y_pred = np.zeros(200)
    while 1:
        X_next = np.zeros((10,200))
        loss = 0
        temp = 0
        for i in range(10):
            X_next[i] = np.linalg.inv(A[i].T @ A[i] + C* np.eye(200, 200))@(C*Y_pred+A[i].T@B[i]-v[i])        
            temp += v[i] + C*X_next[i]        
        Y_next = np.where(temp > lamb, (temp - lamb)/(10*C), np.where(temp < -lamb, (temp + lamb)/(10*C), 0))
        print(np.linalg.norm(Y_pred - Y_next, ord=2))
        if np.linalg.norm(Y_pred - Y_next, ord=2) < 10e-5:break
        for i in range(10):
            v[i] += C*(X_next[i]-Y_next)
        X_pred = X_next
        Y_pred = Y_next
        distance_real.append(np.linalg.norm(Y_next - X_true, ord=2))
        distance_best.append(Y_next)
    draw(distance_real,distance_best)

# 次梯度法
    distance_real,distance_best = [],[]
    alpha = 0.001
    X_pred = np.zeros(200)
    while 1:
        delta_X = 0
        loss = 0
        for i in range(10):
            pred_B = A[i] @ X_pred
            delta_X += (pred_B - B[i]) @ A[i]
        X_next = X_pred - alpha*(delta_X + lamb * np.sign(X_pred))
        if np.linalg.norm(X_pred - X_next, ord=2) < 10e-5: break
        X_pred = X_next
        distance_real.append(np.linalg.norm(X_next - X_true, ord=2))
        distance_best.append(X_next)
    draw(distance_real,distance_best)
