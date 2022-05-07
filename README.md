# 基于动态时间弯曲的多元时间序列模糊聚类

# Dynamic Time Warping based Fuzzy Clustering for Multivariate Time Series

“基于动态时间弯曲的多元时间序列模糊聚类” 的pythton实现

- data

  AUSLAN~uWave：15个多元时间序列数据集

- src

  RI_experiment：在15个数据集上进行聚类的代码（最优参数下）

  utils：WDTW-FCM，DPC，DTW的实现代码

## 依赖

- Python 3
- numba
- numpy
- scipy

## 数据集

- BAYDOGAN M G. Multivariate time series classification datasets[Z]. 2019. http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html

  | 数据集                | 维度 | 样本长度   | 类别数目 | 样本数目 |
  | :-------------------- | :--- | :--------- | :------- | :------- |
  | ArabicDigits          | 13   | 4 ~ 93     | 10       | 2200     |
  | AUSLAN                | 22   | 45 ~ 136   | 95       | 1425     |
  | CharacterTrajectories | 3    | 109 ~ 205  | 20       | 2558     |
  | CMUsubject16          | 62   | 127 ~ 580  | 2        | 29       |
  | ECG                   | 2    | 39 ~ 152   | 2        | 100      |
  | JapaneseVowels        | 12   | 7 ~ 29     | 9        | 370      |
  | Libras                | 2    | 45         | 15       | 180      |
  | uWave                 | 3    | 315        | 8        | 200      |
  | Pendigits             | 2    | 8          | 10       | 300      |
  | WalkvsRun             | 62   | 128 ~ 1918 | 2        | 28       |
  | LP1                   | 6    | 15         | 4        | 50       |
  | LP2                   | 6    | 15         | 5        | 30       |
  | LP3                   | 6    | 15         | 4        | 75       |
  | LP4                   | 6    | 15         | 3        | 75       |
  | LP5                   | 6    | 15         | 5        | 100      |

  LIbras，uWave：Bayesian Learning from Sequential Data using Gaussian Processes with Signature Covariances，https://github.com/tgcsaba/GPSig

  其他：Multivariate LSTM-FCNs for time series classification，https://github.com/houshd/MLSTM-FCN


## 运行模型

运行RI_experiment中WalkvsRun.py~uwave.py来对15个数据集进行聚类

## 结果

| 数据集     | *m*  | *q*  | WDTW  -FCM |   FCFW   | DTW-PAM | DTW-FCMDD | PDC  |    CD    | GAK-DBA | soft-DTW |
| :--------- | :--- | :--: | :--------: | :------: | :-----: | :-------: | ---- | :------: | :-----: | :------: |
| Arabic.    | 1.1  |  4   |  **0.94**  |   0.93   |  0.76   |   0.67    | 0.11 |   0.91   |  0.85   |   0.89   |
| AUSLAN     | 1.4  |  -4  |  **0.99**  |   0.98   |  0.98   | **0.99**  | 0.41 |   0.96   |  0.98   | **0.99** |
| Character. | 1.4  |  -2  |  **0.98**  |   0.96   |  0.94   |   0.95    | 0.85 |   0.89   |  0.96   |   0.93   |
| CMU.       | 1.4  |  -8  |  **1.00**  |   0.66   |  0.66   |   0.66    | 0.5  |   0.85   |  0.62   |   0.50   |
| EC         | 2.0  |  2   |    0.64    | **0.67** |  0.56   |   0.50    | 0.55 |   0.49   |  0.62   |   0.59   |
| Japanese.  | 1.1  | -10  |  **0.98**  |   0.91   |  0.83   |   0.87    | 0.27 |   0.95   |  0.89   |   0.96   |
| Libras     | 1.7  |  4   |  **0.92**  | **0.92** |  0.91   |   0.89    | 0.84 |   0.91   |  0.90   |   0.91   |
| uWave      | 1.1  | -10  |    0.94    |   0.92   |  0.85   |   0.88    | 0.79 | **0.95** |  0.85   |   0.88   |
| Pendigits  | 1.4  |  -8  |  **0.96**  |   0.92   |  0.90   |   0.86    | 0.41 | **0.96** |  0.91   |   0.91   |
| Walkvs.    | 2.0  |  6   |  **1.00**  | **1.00** |  0.48   |   0.60    | 0.54 |   0.71   |  0.54   |   0.48   |
| LP1        | 1.4  |  2   |    0.63    | **0.83** |  0.55   |   0.51    | 0.54 |   0.74   |  0.62   |   0.65   |
| LP2        | 1.1  | -10  |    0.76    | **0.78** |  0.62   |   0.61    | 0.35 |   0.64   |  0.65   |   0.65   |
| LP3        | 2.0  |  4   |  **0.77**  | **0.77** |  0.57   |   0.54    | 0.62 |   0.59   |  0.58   |   0.61   |
| LP4        | 2.0  |  2   |    0.72    | **0.75** |  0.60   |   0.59    | 0.48 |   0.57   |  0.67   |   0.70   |
| LP5        | 2.0  |  2   |    0.63    |   0.74   |  0.52   |   0.72    | 0.46 | **0.76** |  0.60   |   0.45   |
| 平均排名   |      |      |  **1.67**  |   2.00   |  5.47   |   5.47    | 7.13 |   3.87   |  4.53   |   4.20   |
| 胜率       |      |      |  **0.60**  |   0.47   |  0.00   |   0.07    | 0.00 |   0.20   |  0.00   |   0.07   |

FCFW，DTW-PAM，DTW-FCMDD，PDC，CD，GAK-DBA，soft-DTW的实验结果来源于：

LI H, WEI M. Fuzzy clustering based on feature weights for multivariate time series[J]. Knowledge-Based Systems, 2020, 197: 105907.

------

苑瀚洋，浙江大学软件学院，yuanhanyang@zju.edu.cn
