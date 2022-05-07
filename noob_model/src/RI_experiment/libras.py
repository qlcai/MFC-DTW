import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/Libras/LIBRAS.mat"
raw_data = loadmat(path)
X_test = raw_data["X_test"]
X_train = raw_data["X_train"]
Y_test = raw_data["y_test"]
label = Y_test.reshape(-1)
# 归一化处理
normal = None
for i in range(X_test.shape[0]):
    tmp = X_test[i]
    if normal is None:
        normal = tmp
    else:
        normal = np.vstack((normal, tmp))
for i in range(X_train.shape[0]):
    tmp = X_train[i]
    if normal is None:
        normal = tmp
    else:
        normal = np.vstack((normal, tmp))
mean = np.mean(normal, axis=0)
std = np.std(normal, axis=0)
std[np.where(std == 0)] = 1
normal = normal - mean
normal = normal / std

data = []
for i in range(X_test.shape[0]):
    tmp = normal[i * 45: i * 45 + 45, :].T
    data.append(tmp)
print("LIBRAS clustering")
# 设置参数
c = 15
m = 1.7
q = 4
dc_percent = 1.0
max_iter = 20
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('LIBRAS_ri.npy', result)
