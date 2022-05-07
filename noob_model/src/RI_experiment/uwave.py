import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/uWave/UWave.mat"
raw_data = loadmat(path)
raw_data = raw_data['mts']
raw_data = raw_data[0, 0]
X_train = raw_data[1]
Y_train = raw_data[0]
label = Y_train.reshape(-1)
data = []
for i in range(X_train.shape[1]):
    tmp = X_train[0, i]
    data.append(tmp)
print("UWave clustering")
# 设置参数
c = 8
m = 1.1
q = -10
dc_percent = 10.0
max_iter = 20
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('UWave_ri.npy', result)
