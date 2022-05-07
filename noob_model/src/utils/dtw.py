import numpy as np
from numba import jit


@jit
def get_dtw(t1, t2, lam, q):
    """
    计算WDTW的路径和距离
    :param t1: ndarray, D * b
    :param t2: ndarray, D * a
    :param lam: lam
    :param q: q
    """
    # t1, t2的长度
    b, a = t1.shape[1], t2.shape[1]
    # 初始化点对距离矩阵，动态规划矩阵
    dis_mat = np.zeros((b, a))
    dp_mat = np.zeros((b, a))
    # 计算点对距离矩阵，动态规划矩阵
    for i in range(b):
        for j in range(a):
            # 计算t1中第i个元素与t2中第j个元素的点对距离
            dis_mat[i, j] = dist_fun(t1[:, i], t2[:, j], lam, q)
            if i == 0 and j == 0:
                dp_mat[i, j] = dis_mat[i, j]
            elif i == 0:
                dp_mat[i, j] = dp_mat[i, j - 1] + dis_mat[i, j]
            elif j == 0:
                dp_mat[i, j] = dp_mat[i - 1, j] + dis_mat[i, j]
            else:
                min_list = [dp_mat[i - 1, j - 1], dp_mat[i - 1, j], dp_mat[i, j - 1]]
                dp_mat[i, j] = dis_mat[i, j] + min(min_list)

    # 计算owp
    path = traceback(dp_mat)
    return dp_mat[-1, -1], path


# 通过加权欧式距离计算点对距离
@jit
def dist_fun(t1_i, t2_j, lam, q):
    dist = np.power(t1_i - t2_j, 2)
    weighted_dist = np.multiply(np.power(lam, q), dist)
    return np.sum(weighted_dist)


# 找到owp
@jit
def traceback(dp_mat):
    # 分别存储owp中t1和t2对应的下标
    p_1, p_2 = [], []
    # 从动态规划矩阵的终点开始回溯
    i, j = dp_mat.shape[0]-1, dp_mat.shape[1]-1
    while (i > 0) or (j > 0):
        p_1.insert(0, i)
        p_2.insert(0, j)
        if i == 0:
            tb = 1
        elif j == 0:
            tb = 2
        else:
            # tb = np.argmin((dp_mat[i - 1, j - 1], dp_mat[i, j - 1], dp_mat[i - 1, j]))
            min_list = [dp_mat[i - 1, j - 1], dp_mat[i, j - 1], dp_mat[i - 1, j]]
            min_value = min(min_list)
            if min_list[0] == min_value:
                tb = 0
            elif min_list[1] == min_value:
                tb = 1
            else:
                tb = 2

        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            j -= 1
        else:
            # (tb == 2):
            i -= 1
    # 将（0，0）存入路径
    p_1.insert(0, i)
    p_2.insert(0, j)
    # 将p_1,p_2拼接为一个2*l的矩阵（l为owp长度）
    return np.vstack((np.array(p_1), np.array(p_2)))
