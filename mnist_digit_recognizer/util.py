#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:29:51 2018

@author: aguiroz
"""

import numpy as np

def loadData():
    data = np.loadtxt(open('data/train.csv'), delimiter=',', skiprows=1, dtype=np.float64)
    x = np.array([i[1:] for i in data])
    y = np.array([i[0] for i in data])
    return x,y

def getIndicator(y: np.array):
    y = y.astype(np.int32)
    n = len(y)
    ind = np.zeros((n, 10))
    for i in range(n):
        ind[i, y[i]] = 1
    return ind