import numpy as np
from src.utils import dtw
from src.utils import dpc
import copy
import random
import time
FLOAT_MAX = 1e100


class WDtwFcm:
    def __init__(self, data, c, m, q, max_iter, label, dc_percent):
        self.max_iter = max_iter  # 最大迭代次数
        self.x = data  # 数据集，list类型， n*D*a
        self.c = c  # 类别数
        self.m = m  # 模糊参数
        self.q = q  # 权重参数
        self.label = label  # 样本标签
        self.dc_percent = dc_percent  # dpc截断距离的百分比

        self.D = self.x[0].shape[0]  # 维度
        self.n = len(self.x)  # 样本数

        # 分别存储簇到样本之间的dtw路径和距离
        self.dtw_path = []
        self.dtw_dis = np.ones((self.c, self.n)) * FLOAT_MAX
        # 初始化u，lam，v以及目标函数值
        self.u = np.ones((self.c, self.n)) / self.c
        self.lam = np.ones(self.D) / self.D
        self.v = None
        self.obj_func = FLOAT_MAX
        print("data info: dimension:", self.D, " class:", self.c, " volume:", self.n)

# 通过DPC初始化簇中心
    def dpc_initiate(self):
        dpc_cen_index = dpc.get_dpc(self.x, lam=self.lam, c=self.c, percent=self.dc_percent)
        cen_cen = []
        for index in dpc_cen_index:
            cen_cen.append(self.x[index])
        return cen_cen

# 更新簇到样本之间的WDTW路径和距离
    def update_dtw(self):
        new_dtw_dis = np.ones((self.c, self.n)) * FLOAT_MAX
        new_dtw_path = []
        for i in range(self.c):
            tmp = []
            for j in range(self.n):
                new_dtw_dis[i, j], path = dtw.get_dtw(t1=self.v[i], t2=self.x[j], lam=self.lam, q=self.q)
                tmp.append(path)
            new_dtw_path.append(tmp)
        return new_dtw_dis, new_dtw_path

# 更新隶属度矩阵U
    def update_u(self):
        # 初始化新的u
        new_u = np.ones((self.c, self.n)) * FLOAT_MAX
        # 根据公式（7）更新u
        for i in range(self.c):
            for j in range(self.n):
                lower_sum = 0
                is_coincide = [False, 0]
                for s in range(self.c):
                    # 判断x_j和某一簇中心的距离是否为0
                    if self.dtw_dis[s, j] == 0:
                        is_coincide[0] = True
                        is_coincide[1] = s
                        break
                    lower_sum += pow(self.dtw_dis[i, j] / self.dtw_dis[s, j], 1 / (self.m - 1))
                # 若存在距离为0的情况，则u_ij只能为1或0
                if is_coincide[0]:
                    if is_coincide[1] == i:
                        new_u[i, j] = 1
                    else:
                        new_u[i, j] = 0
                else:
                    new_u[i, j] = 1 / lower_sum
        return new_u

# 更新权重lam
    def update_lam(self):
        # 先计算每一维度下公式（10）中的sum_ijk一项
        tmp = []
        for s in range(self.D):
            tmp2 = 0
            for i in range(self.c):
                for j in range(self.n):
                    # v_i与x_j对应的owp，是2*l维的矩阵（l是owp长度）
                    path = self.dtw_path[i][j]
                    for k in range(path.shape[1]):
                        # path[0, k]表示公式（10）中的p_k_1，path[1, k]表示p_k_2
                        tmp2 += pow(self.u[i, j], self.m) * \
                                pow(self.v[i][s, path[0, k]] - self.x[j][s, path[1, k]], 2)
            tmp.append(tmp2)
        # 初始化新的lam
        new_lam = np.ones(self.D) * FLOAT_MAX
        # 根据公式（10）更新lam
        for d in range(self.D):
            lower_sum = 0
            for s in range(self.D):
                lower_sum += pow(tmp[d] / tmp[s], 1 / (self.q - 1))
            new_lam[d] = 1 / lower_sum
        return new_lam

# 更新簇中心
    def update_v(self):
        # 新的簇中心集合
        new_v = []
        for i in range(self.c):
            # v_i的长度
            b = self.v[i].shape[1]
            # 公式（13）中分子部分和分母部分
            upper_sum = np.zeros((self.D, b))
            lower_sum = np.zeros(b)
            for j in range(self.n):
                # path是v_i与x_j对应的owp，2*l维矩阵
                path = self.dtw_path[i][j]
                x_j = self.x[j]
                # 遍历owp
                for k in range(path.shape[1]):
                    # 根据owp可以判断，x_j中的第path[1, k]列对v_i中的第path[0, k]列做贡献
                    # 因此累加在upper_sum和lower_sum的第path[0, k]列上
                    upper_sum[:, path[0, k]] += x_j[:, path[1, k]] * pow(self.u[i, j], self.m)
                    lower_sum[path[0, k]] += pow(self.u[i, j], self.m)
            # upper_sum与lower_sum的对应列相除得到新的v_i
            new_v_i = upper_sum / lower_sum
            new_v.append(new_v_i)
        return new_v

# 更新目标函数
    def update_obj_func(self):
        new_obj_func = 0
        # 根据公式（5）计算目标函数值
        for i in range(self.c):
            v_i = self.v[i]
            for j in range(self.n):
                x_j = self.x[j]
                path = self.dtw_path[i][j]
                for k in range(path.shape[1]):
                    dist = np.power(v_i[:, path[0, k]] - x_j[:, path[1, k]], 2)
                    # 每一维度分别乘权重系数后加总
                    sum_weighted_dist = np.sum(np.multiply(np.power(self.lam, self.q), dist))
                    new_obj_func += sum_weighted_dist * pow(self.u[i, j], self.m)
        return new_obj_func

# WDTW—FCM的实现
    def get_w_dtw_fcm(self):
        start = time.clock()
        # 初始化簇中心
        self.v = self.dpc_initiate()
        for i in range(self.max_iter):
            # print(i)
            # 更新dtw路径和距离，u，lam，v以及目标函数值
            self.dtw_dis, self.dtw_path = self.update_dtw()
            self.u = self.update_u()
            self.lam = self.update_lam()
            self.v = self.update_v()
            tmp = self.update_obj_func()
            # 超出精度范围
            if tmp > self.obj_func:
                break
            self.obj_func = tmp
            # print(self.obj_func)
        end = time.clock()
        print("model run time: ", (end - start) / 60, "min")
        print("best lam:", self.lam)
        return self.cal_ri(np.argmax(self.u, axis=0))

# 计算Rand指数
    def cal_ri(self, y_pred):
        # 计算ri指数
        n = len(self.label)
        a, b = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if (self.label[i] == self.label[j]) & (y_pred[i] == y_pred[j]):
                    a += 1
                elif (self.label[i] != self.label[j]) & (y_pred[i] != y_pred[j]):
                    b += 1
                else:
                    pass
        ri = (a + b) / (n * (n - 1) / 2)
        return ri
