import numpy as np
import matplotlib.pyplot as plt
from src.utils import dtw
from scipy.io import loadmat
import time


# 计算数据点两两之间的距离
def getDistanceMatrix(data, lam):
    n = len(data)
    dist_mat = np.ones((n, n)) * 1e100
    for i in range(n):
        for j in range(i, n):
            dist_mat[i, j] = dist_mat[j, i] = dtw.get_dtw(t1=data[i], t2=data[j], lam=lam, q=1)[0]
    return dist_mat


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的percent%
def select_dc(dists, percent):
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    for i in range(N):
        if method is None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离,即对每个点，找到密度比它大的所有点,再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            deltas[index_rho[0]] = np.max(dists[index, :])
            continue
        # 对于其他的点,找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])
    return deltas


# 选取rho与delta乘积较大的点作为聚类中心
def find_centers_c(rho, deltas, c):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:c]


# DPC初始化簇中心的实现
def get_dpc(data, lam, c, percent):
    # 计算距离矩阵
    dists = getDistanceMatrix(data=data, lam=lam)
    dc = select_dc(dists, percent=percent)
    # 计算局部密度
    # rho = get_density(dists, dc, method="exp")
    rho = get_density(dists, dc)
    # 计算密度距离
    deltas = get_deltas(dists, rho)
    # 获取聚类中心点
    centers = find_centers_c(rho, deltas, c=c)
    print("dpc centers:", centers)
    return centers

