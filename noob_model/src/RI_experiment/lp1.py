import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/LP1/LP1.mat"
raw_data = loadmat(path)
X_test = raw_data["X_test"]
Y_test = raw_data["Y_test"]
label = Y_test.reshape(-1)
data = []
for i in range(X_test.shape[1]):
    tmp = X_test[0, i]
    data.append(tmp)
print("LP1 clustering")
# 设置参数
c = 4
m = 1.4
q = 2
dc_percent = 10.0
max_iter = 40
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('LP1_ri.npy', result)


