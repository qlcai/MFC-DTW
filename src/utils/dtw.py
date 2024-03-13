import numpy as np


def get_dtw(t1, t2, lamda, q):
    """
    Compute DTW distance and OWPs
    :param t1: ndarray, D * b
    :param t2: ndarray, D * a
    :param lamda: weights
    :param q: weight order
    :return: DTW distance and OWPs
    """
    b, a = t1.shape[1], t2.shape[1]
    dis_mat = np.zeros((b, a))
    dp_mat = np.zeros((b, a))

    for i in range(b):
        for j in range(a):
            dis_mat[i, j] = dist_fun(t1[:, i], t2[:, j], lamda, q)
            if i == 0 and j == 0:
                dp_mat[i, j] = dis_mat[i, j]
            elif i == 0:
                dp_mat[i, j] = dp_mat[i, j - 1] + dis_mat[i, j]
            elif j == 0:
                dp_mat[i, j] = dp_mat[i - 1, j] + dis_mat[i, j]
            else:
                min_list = [dp_mat[i - 1, j - 1], dp_mat[i - 1, j], dp_mat[i, j - 1]]
                dp_mat[i, j] = dis_mat[i, j] + min(min_list)

    path = traceback(dp_mat)
    return dp_mat[-1, -1], path


def dist_fun(t1_i, t2_j, lamda, q):
    """
    Wighted Euclidean distance
    """
    dist = np.power(t1_i - t2_j, 2)
    weighted_dist = np.multiply(np.power(lamda, q), dist)
    return np.sum(weighted_dist)


def traceback(dp_mat):
    """
    Get OWP
    """
    p_1, p_2 = [], []
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
            i -= 1

    p_1.insert(0, i)
    p_2.insert(0, j)

    return np.vstack((np.array(p_1), np.array(p_2)))
