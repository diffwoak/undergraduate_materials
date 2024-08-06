import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 划分80%训练集和20%测试集
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

# 构建用户的movie交集数量表
def gen_cross_nums(mat):
    cross_nums = pd.DataFrame(0.0, index = mat.index, columns = mat.index)
    for column_name, column_data in mat.items():
        non_nan_column_data = column_data.dropna()
        for index_1, value_1 in non_nan_column_data.items():
            for index_2, value_2 in non_nan_column_data.items():
                if index_1 == index_2:
                    continue
                cross_nums.loc[index_1][index_2] += value_1 * value_2
    cross_nums.to_csv('tmp-matrix/cross_nums.csv')
    return cross_nums

# 构建用户的movie数量表
def gen_movie_nums(mat):
    movie_nums = pd.DataFrame(0.0, index=mat.index, columns=['movie_nums'])
    for index, row_data in mat.iterrows():
        row_sum = row_data.sum(skipna=True)
        movie_nums.loc[index][0] = row_sum
    movie_nums.to_csv('tmp-matrix/movie_nums.csv')
    return movie_nums

# 构建用户的相似度表
def gen_user_sim(mat,cross_nums,movie_nums):
    user_sim = pd.DataFrame(0.0, index=mat.index, columns=mat.index)
    for user1_id, users2 in cross_nums.items():
        for user2_id, internums in users2.items():
            user1_id = int(user1_id)
            user2_id = int(user2_id)
            user_sim.loc[user1_id][user2_id] = internums / pow(movie_nums.loc[user1_id][0] * movie_nums.loc[user2_id][0],0.5)
    user_sim.to_csv('tmp-matrix/user_sim.csv')      
    return user_sim

# 生成baseline estimate
def gen_base_est(mat):
    means = mat.sum(skipna=True).sum()/mat.count().sum()
    b_user = mat.mean(axis=1,skipna=True) - means
    b_movie = mat.mean(skipna=True) - means
    base_est = pd.DataFrame(means, index=mat.index, columns=mat.columns)
    for user_id in mat.index:
        for movie_id in mat.columns:
            base_est.loc[user_id][movie_id] += b_user[user_id] + b_movie[movie_id]
    base_est.to_csv('tmp-matrix/base_est.csv')  
    return base_est

# 计算user_id movie_id所对应的rate值
def calcu_rate(user_id,movie_id,k,train_matrix,user_sim,base_est):
    train_matrix_filled = train_matrix.fillna(0)
    if train_matrix_filled.loc[user_id][str(movie_id)] > 0:
        return train_matrix.loc[user_id][str(movie_id)]
    sims = user_sim.loc[user_id]
    top_k_sims = sims.sort_values(ascending=False).head(k)
    top_k_users = top_k_sims.index.tolist()
    pred_rate = 0
    sim_sum = 0
    for related_user in top_k_users:
        pred_rate += user_sim.loc[user_id][related_user] * (train_matrix_filled.loc[int(related_user)][str(movie_id)] - base_est.loc[int(related_user)][str(movie_id)])
        sim_sum += user_sim.loc[user_id][related_user]
    pred_rate /= sim_sum
    pred_rate += base_est.loc[user_id][str(movie_id)]
    return pred_rate

# 计算RMSE误差
def RMSE(train_matrix,test_matrix,user_sim,base_est):
    k = 10
    cost = 0
    for user_id in test_matrix.index:
        for movie_id in test_matrix.columns:
            expect = test_matrix.at[user_id, movie_id]
            if expect > 0:
                calcu = calcu_rate(user_id,movie_id,k,train_matrix,user_sim,base_est)
                cost += (calcu - expect) ** 2
                # print(f"calcu: {calcu} , expect: {expect}")
    return pow(cost/(test_matrix.count().sum()),0.5)

if __name__ == "__main__":

    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    rating_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    # 划分训练集和测试集
    # 划分过程比较耗时,故将一次的划分结果保存文件,在实验过程中直接读取文件
    train_matrix,test_matrix = get_TrainTest_data(rating_matrix)
    train_matrix = pd.read_csv('tmp-matrix/train_matrix.csv', index_col=0)
    test_matrix = pd.read_csv('tmp-matrix/test_matrix.csv', index_col=0)

    print("Original matrix sample counts:", rating_matrix.count().sum())
    print("Train matrix sample counts:", train_matrix.count().sum())
    print("Test matrix sample counts:", test_matrix.count().sum())
    
    # 1. 构建用户的movie交集数量表
    # cross_nums = gen_cross_nums(train_matrix)
    cross_nums = pd.read_csv('tmp-matrix/cross_nums.csv', index_col=0)
    print('read corss nums..')
    # 2. 构建用户的movie数量表
    # movie_nums = gen_movie_nums(train_matrix)
    movie_nums = pd.read_csv('tmp-matrix/movie_nums.csv', index_col=0)
    print('read movie nums..')
    # 3. 构建用户的相似度表
    # user_sim = gen_user_sim(rating_matrix,cross_nums,movie_nums)
    user_sim = pd.read_csv('tmp-matrix/user_sim.csv', index_col=0)
    print('read user sim..')

    # 4. 计算baseline estimate表
    # base_est = gen_base_est(rating_matrix)
    base_est = pd.read_csv('tmp-matrix/base_est.csv', index_col=0)
    print('read base est..')
    
    # 使用生成的表计算rate值,计算RMSE误差
    loss = RMSE(train_matrix,test_matrix,user_sim,base_est)
    print(f"RMSE: {loss}")