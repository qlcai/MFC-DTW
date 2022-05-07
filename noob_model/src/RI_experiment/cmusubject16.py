import sys
sys.path.append("../..")
import numpy as np
from src.utils import model
from scipy.io import loadmat
# 读取数据
path = "../../data/CMUsubject16/CMUsubject16.mat"
raw_data = loadmat(path)
X_test = raw_data["X_test"]
Y_test = raw_data["Y_test"]
label = Y_test.reshape(-1)
data = []
for i in range(X_test.shape[1]):
    tmp = X_test[0, i]
    data.append(tmp)
print("CMUsubject16 clustering")
# 设置参数
c = 2
m = 1.4
q = -8
dc_percent = 10.0
max_iter = 20
print("parameters:", " m:", m, " q:", q, " dc_percent", dc_percent, " max iter:", max_iter)
# 运行模型
opt = model.WDtwFcm(data=data, c=c, m=m, q=q, max_iter=max_iter, label=label, dc_percent=dc_percent)
result = opt.get_w_dtw_fcm()
print("rand index:", result)
# np.save('CMUsubject16_ri.npy', result)
