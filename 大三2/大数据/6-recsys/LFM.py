import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_TrainTest_data(mat):
    train_matrix = pd.DataFrame(np.nan, index=mat.index, columns=mat.columns)
    test_matrix = pd.DataFrame(np.nan, index=mat.index, columns=mat.columns)
    for user_id in mat.index:
        non_nan_indices = mat.loc[user_id].dropna().index.tolist()
        train_indices, test_indices = train_test_split(non_nan_indices, test_size=0.2, random_state=43)
        train_matrix.loc[user_id, train_indices] = mat.loc[user_id, train_indices]
        test_matrix.loc[user_id, test_indices] = mat.loc[user_id, test_indices]
    train_matrix.to_csv('tmp-matrix/train_matrix.csv')
    test_matrix.to_csv('tmp-matrix/test_matrix.csv')
    return train_matrix, test_matrix

def LFM(train_matrix,test_matrix,K,max_iter,lr,reg):
    M, N = train_matrix.shape
    U = np.random.normal(scale=1./K, size=(M, K))
    V = np.random.normal(scale=1./K, size=(N, K))
    for step in range(max_iter):
        cost = 0
        for u in range(M):
            for i in range(N):
                if train_matrix[u][i] > 0:
                    err = train_matrix[u][i] - np.dot(U[u,:],V[i,:])
                    cost +=  err ** 2
                    U[u, :] += lr * (err * V[i, :] - reg * U[u, :])
                    V[i, :] += lr * (err * U[u, :] - reg * V[i, :])
        if step % 10 == 0:
            loss = RMSE(np.dot(U,V.T),test_matrix)
            print(f"step: {step}, cost: {cost}, train_RMSE: {pow(cost/np.count_nonzero(train_matrix),0.5)}, test_RMSE:{loss}")           
        if pow(cost/np.count_nonzero(train_matrix),0.5) < 0.01:
            break
    return np.dot(U,V.T)

def RMSE(pred_R,test_matrix):
    M, N = pred_R.shape
    cost = 0
    for u in range(M):
        for i in range(N):
            if test_matrix[u][i] > 0:
                cost += (pred_R[u][i] - test_matrix[u][i]) ** 2
    return pow(cost/np.count_nonzero(test_matrix),0.5)

if __name__ == "__main__":
    K= 20
    max_iter = 200
    lr = 0.01
    reg = 0.1

    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    rating_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    # 划分训练集和测试集
    # 划分过程比较耗时,故将一次的划分结果保存文件,在实验过程中直接读取文件
    # train_matrix,test_matrix = get_TrainTest_data(rating_matrix)
    train_matrix = pd.read_csv('tmp-matrix/train_matrix.csv', index_col=0)
    test_matrix = pd.read_csv('tmp-matrix/test_matrix.csv', index_col=0)

    print("Original matrix sample counts:", rating_matrix.count().sum())
    print("Train matrix sample counts:", train_matrix.count().sum())
    print("Test matrix sample counts:", test_matrix.count().sum())

    train_matrix = train_matrix.to_numpy()
    train_matrix = np.nan_to_num(train_matrix)
    test_matrix = test_matrix.to_numpy()
    test_matrix = np.nan_to_num(test_matrix)

    pred_R = LFM(train_matrix,test_matrix,K,max_iter,lr,reg)
    # np.savetxt('pred_R.csv', pred_R, delimiter=',')
    loss = RMSE(pred_R,test_matrix)
    print(f"RMSE: {loss}")