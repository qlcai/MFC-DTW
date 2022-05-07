import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/Pendigits/pendigits.mat"
raw_data = loadmat(path)
X_train = raw_data["X_train"]
Y_train = raw_data["Y_train"]
label = Y_train.reshape(-1)
data = []
for i in range(X_train.shape[1]):
    tmp = X_train[0, i]
    data.append(tmp)
print("pendigits clustering")
# 设置参数
c = 10
m = 1.4
q = -8
dc_percent = 2.0
max_iter = 20
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('pendigits_ri.npy', result)
