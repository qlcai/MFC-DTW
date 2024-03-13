## Data-Adaptive Dynamic Time Warping Based Multivariate Time Series Fuzzy Clustering

## Files
- data

  8 MTS datasets in the format of matlab file
  
- src

  parameter.json: the parameters (class number, exponents of fuzzy coefficients and weights) for each dataset
  utils: MFC-DTW implementation files


## Datasets

- BAYDOGAN M G. Multivariate time series classification datasets. http://www.mustafabaydogan.com

  | Dataset               | Dim. | Length     | Class    | Size     |
  | :-------------------- | :--- | :--------- | :------- | :------- |
  | ArabicDigits          | 13   | 4 ~ 93     | 10       | 2200     |
  | AUSLAN                | 22   | 45 ~ 136   | 95       | 1425     |
  | CharacterTrajectories | 3    | 109 ~ 205  | 20       | 2558     |
  | CMUsubject16          | 62   | 127 ~ 580  | 2        | 29       |
  | ECG                   | 2    | 39 ~ 152   | 2        | 100      |
  | JapaneseVowels        | 12   | 7 ~ 29     | 9        | 370      |
  | uWave                 | 3    | 315        | 8        | 200      |
  | LP4                   | 6    | 15         | 3        | 75       |

- Source

  uWave: Bayesian Learning from Sequential Data using Gaussian Processes with Signature Covariances，https://github.com/tgcsaba/GPSig

  Others: Multivariate LSTM-FCNs for time series classification，https://github.com/houshd/MLSTM-FCN

------

