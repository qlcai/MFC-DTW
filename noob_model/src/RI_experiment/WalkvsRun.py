import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/WalkvsRun/WalkvsRun.mat"
raw_data = loadmat(path)
X_train = raw_data["X_train"]
Y_train = raw_data["Y_train"]
label = Y_train.reshape(-1)
data = []
for i in range(X_train.shape[1]):
    tmp = X_train[0, i]
    data.append(tmp)
print("WalkvsRun clustering")
# 设置参数
c = 2
m = 2.0
q = 6
dc_percent = 50.0
max_iter = 20
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('WalkvsRun_ri.npy', result)
