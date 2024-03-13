import time

import numpy as np
import math
from src.utils import dtw
from src.utils import dpc


class MfcDtw:

    def __init__(self, data, c, m, q, max_iter, dc_percent, class_label=None, anom_label=None):
        self.max_iter = max_iter                        # maximum iterations
        self.x = data                                   # input data
        self.c = c                                      # class number
        self.m = m                                      # fuzzy order
        self.q = q                                      # weight order
        self.class_label = class_label                  # class label
        self.dc_percent = dc_percent                    # intercept percentage of DPC
        self.D = self.x[0].shape[0]                     # sample dimensions
        self.n = len(self.x)                            # dataset size
        self.dtw_path = []                              # OWP
        self.dtw_dist = np.zeros((self.c, self.n))      # DTW distance
        self.u = np.ones((self.c, self.n)) / self.c     # membership degree matrix
        self.lamda = np.ones(self.D) / self.D           # dimension weights
        self.v = None                                   # cluster centers
        self.loss = math.inf                            # loss of objective function
        print("dataset info: dimension:", self.D, " class:", self.c, " size:", self.n)

    def dpc_initiate(self):
        """
        Initialize cluster centers
        """
        dpc_centers = dpc.get_dpc(self.x, lamda=self.lamda, c=self.c, percent=self.dc_percent)
        centers = []
        for ind in dpc_centers:
            centers.append(self.x[ind])
        return centers

    def update_dtw(self):
        """
        Update DTW distance and OWPs
        """
        new_dist = np.zeros((self.c, self.n))
        new_owp = []
        for i in range(self.c):
            tmp = []
            for j in range(self.n):
                new_dist[i, j], path = dtw.get_dtw(t1=self.v[i], t2=self.x[j], lamda=self.lamda, q=self.q)
                tmp.append(path)
            new_owp.append(tmp)
        return new_dist, new_owp

    def update_u(self):
        """
        Update membership degree matrix
        """
        new_u = np.zeros((self.c, self.n))
        for i in range(self.c):
            for j in range(self.n):
                denom_sum = 0
                is_coincide = [False, 0]

                for s in range(self.c):
                    if self.dtw_dist[s, j] == 0:
                        is_coincide[0] = True
                        is_coincide[1] = s
                        break
                    denom_sum += pow(self.dtw_dist[i, j] / self.dtw_dist[s, j], 1 / (self.m - 1))

                if is_coincide[0]:
                    if is_coincide[1] == i:
                        new_u[i, j] = 1
                    else:
                        new_u[i, j] = 0
                else:
                    new_u[i, j] = 1 / denom_sum
        return new_u

    def update_lamda(self):
        """
        Update dimension weights
        """
        A = []
        for s in range(self.D):
            Ad = 0
            for i in range(self.c):
                for j in range(self.n):
                    path = self.dtw_path[i][j]
                    for k in range(path.shape[1]):
                        if self.u[i, j] == 0:
                            self.u[i, j] = 0.0001
                        Ad += pow(self.u[i, j], self.m) * \
                              pow(self.v[i][s, path[0, k]] - self.x[j][s, path[1, k]], 2)
            A.append(Ad)

        new_lamda = np.zeros(self.D)
        for d in range(self.D):
            denom_sum = 0
            for s in range(self.D):
                if A[d] == 0:
                    A[d] = 0.0001
                denom_sum += pow(A[d] / (A[s] + 1), 1 / (self.q - 1))
            new_lamda[d] = 1 / denom_sum
            if new_lamda[d] > 100 or new_lamda[d] == 0:
                new_lamda[d] = 1e-6

        return new_lamda

    def update_v(self):
        """
        Update cluster center
        """
        new_v = []
        for i in range(self.c):
            b = self.v[i].shape[1]
            numer_sum = np.zeros((self.D, b))
            denom_sum = np.zeros(b)

            for j in range(self.n):
                path = self.dtw_path[i][j]
                x_j = self.x[j]
                for k in range(path.shape[1]):
                    if self.u[i, j] == 0:
                        self.u[i, j] = 0.0001
                    numer_sum[:, path[0, k]] += x_j[:, path[1, k]] * pow(self.u[i, j], self.m)
                    denom_sum[path[0, k]] += pow(self.u[i, j], self.m)

            new_vi = numer_sum / denom_sum
            new_v.append(new_vi)
        return new_v

    def update_loss(self):
        """
        Update the loss of objective function
        """
        new_loss = 0
        for i in range(self.c):
            v_i = self.v[i]
            for j in range(self.n):
                x_j = self.x[j]
                path = self.dtw_path[i][j]
                for k in range(path.shape[1]):
                    dist = np.power(v_i[:, path[0, k]] - x_j[:, path[1, k]], 2)
                    sum_dist = np.sum(np.multiply(np.power(self.lamda, self.q), dist))
                    if self.u[i, j] == 0:
                        self.u[i, j] += 0.0001
                    new_loss += sum_dist * pow(self.u[i, j], self.m)
        return new_loss

    def mfc_dtw(self):
        """
        MFC-DTW realization
        """
        # initialize cluster centers
        self.v = self.dpc_initiate()
        print("Initialize cluster centers")

        start_time = time.time()
        all_loss = []
        for i in range(self.max_iter):
            print("iteration: ", i)
            self.dtw_dist, self.dtw_path = self.update_dtw()  # update DTW OWPs
            print("Update OWP")
            self.u = self.update_u()  # update membership matrix
            print("Update U")
            self.lamda = self.update_lamda()  # update dimension weights
            print("Update lamda")
            print(self.lamda)
            self.v = self.update_v()  # update cluster centers
            print("Update cluster centers")
            print(self.v)
            loss = self.update_loss()  # update loss
            if loss > self.loss:
                break
            self.loss = loss
            all_loss.append(loss)
            print("Opt loss: ", loss)

        end_time = time.time()
        time_cost = end_time - start_time

        ri = None
        if self.class_label is not None:
            ri = self.cal_ri(np.argmax(self.u, axis=0))

        return ri, all_loss, time_cost

    def cal_ri(self, y_pred):
        """
        Compute Rand Index
        """
        n = len(self.class_label)
        a, b = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if (self.class_label[i] == self.class_label[j]) & (y_pred[i] == y_pred[j]):
                    a += 1
                elif (self.class_label[i] != self.class_label[j]) & (y_pred[i] != y_pred[j]):
                    b += 1
                else:
                    pass
        ri = (a + b) / (n * (n - 1) / 2)
        return ri
