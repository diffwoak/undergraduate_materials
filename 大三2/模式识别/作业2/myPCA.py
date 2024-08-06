import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def my_pca(X, k):
    # 中心化数据
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # 计算协方差矩阵
    X_covariance = np.cov(X_centered, rowvar=False)
    # SVD 奇异值分解
    U, _, _ = np.linalg.svd(X_covariance)
    # 提取前 k 个主成分
    coeff = U[:, :k]
    # 计算主成分值
    score = np.dot(X_centered, coeff)
    return coeff, score

def showCoeff(coeff):
    # 展示前49个主成分
    fig, axes = plt.subplots(7, 7, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        print(f"shape:{coeff[:,i].shape}")
        component = coeff[:, i].reshape(32, 32)
        component = np.rot90(component, k=-1, axes=(0, 1))
        ax.imshow(component, cmap='gray')
        ax.axis('off')
    plt.savefig('results/PCA/eigen_faces.jpg')
    plt.show()

def faces():
    mat = scipy.io.loadmat('data/faces.mat')
    X = mat['X']
    nums = X.shape[0]
    ks = [10,50,100,150]
    fig, axes = plt.subplots(5,6, figsize=(8, 8))
    for i, k in enumerate(ks):
        # 压缩
        coeff, score = my_pca(X, k)
        showCoeff(coeff)  # 展示前49个主成分
        # 重建
        re_X = np.dot(score, coeff.T) + np.mean(X, axis=0)
        # 展示
        re_X = re_X.reshape((nums, 32, 32))
        re_X = np.rot90(re_X, k=-1, axes=(1, 2)) # 顺时针旋转90度矫正
        axes[i+1][0].set_title(f'image k = {k}')
        for j in range(6):
            axes[i+1][j].imshow(re_X[j], cmap='gray')
            axes[i+1][j].axis('off')
    X = X.reshape((nums, 32, 32))
    X = np.rot90(X, k=-1, axes=(1, 2)) # 顺时针旋转90度矫正
    axes[0][0].set_title(f'original image')
    for j in range(6):
        axes[0][j].imshow(X[j], cmap='gray')
        axes[0][j].axis('off')
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    # plt.savefig(f'results/PCA/compare_recovered_faces.jpg')
    plt.show()


def scenery():
    image = Image.open('data/scenery.jpg')
    img = np.array(image)
    h,w,_ = img.shape
    # 展开，每个像素视作一个特征
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    ks = [10,50,100,150]
    fig, axes = plt.subplots(1,5, figsize=(10, 5))
    for i, k in enumerate(ks):
        # 压缩
        coeff_R, score_R = my_pca(R, k)
        coeff_G, score_G = my_pca(G, k)
        coeff_B, score_B = my_pca(B, k)
        # 重建
        re_R = np.dot(score_R, coeff_R.T) + np.mean(R, axis=0)
        re_G = np.dot(score_G, coeff_G.T) + np.mean(G, axis=0)
        re_B = np.dot(score_B, coeff_B.T) + np.mean(B, axis=0)
        re_X = np.stack((re_R, re_G, re_B), axis=-1).astype(np.uint8)
        # 展示
        axes[i+1].set_title(f'image k = {k}')
        axes[i+1].imshow(re_X)
        axes[i+1].axis('off')
    axes[0].set_title(f'original image')
    axes[0].imshow(img)
    axes[0].axis('off')
    # plt.savefig(f'results/PCA/compare_recovered_scenery.jpg')
    plt.show()

if __name__ == "__main__":
    faces()
    scenery()

    
    
