#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utils import mfcdtw
from scipy.io import loadmat
import json

with open('parameter.json', 'r') as f:
    data = json.load(f)

dataset = ['ArabicDigits',
           'AUSLAN',
           'CharacterTrajectories',
           'CMUsubject16',
           'ECG',
           'JapaneseVowels',
           'uWave',
           'LP4']

for name in ['LP4']:
    para = data[name]
    path = "../data/" + name + ".mat"
    raw_data = loadmat(path)
    if name == 'uWave':
        suf = 'train'
    else:
        suf = 'test'
    X = raw_data["X_" + suf]
    Y = raw_data["Y_" + suf]
    label = Y.reshape(-1)
    data = list(X[0])
    print("cluster " + name)

    iteration = 20
    opt = mfcdtw.MfcDtw(data=data, c=para['class'], m=para['m'], q=para['q'], max_iter=iteration,
                        dc_percent=para['dc_intercept'], class_label=label)

    result = opt.mfc_dtw()
    print("RI value: ", result[0])
    print("time cost: ", result[2])
