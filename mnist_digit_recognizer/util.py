#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:49:52 2018

@author: aguiroz
"""

import numpy as np
import os

def load_train_data():
    mnist = np.loadtxt("data/train.csv", delimiter=",", skiprows=1, dtype=np.float32)
    
    x_train = np.array([i[1:] / 255 for i in mnist[:2000]])
    y_train = np.array([i[0] for i in mnist[:2000]])

    x_test = np.array([i[1:] / 255 for i in mnist[-1000:]])
    y_test = np.array([i[0] for i in mnist[-1000:]])
    
    return x_train, y_train, x_test, y_test

def get_indicator(y):
    y = y.astype(np.int32)
    n = len(y)
    ind = np.zeros((n, 10))
    for i in range(n):
        ind[i, y[i]] = 1
        
    return ind

def classificationRate(target, prediction):
        n_correct = 0
        for i in range(len(target)):
            if target[i] == prediction[i]:
                n_correct += 1
        return n_correct

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def relu(x):
    return x * (x > 0)

def softmax(x):
    expX = np.exp(x)  
    return expX / expX.sum(axis=1, keepdims=True)

def check_path(name):
    if not os.path.exists("model"):
        os.mkdir("model")
        
    if not os.path.exists("model/{}".format(name)):
        os.mkdir("model/{}".format(name))
    
    return
        
def check_model(name, fw):
    return os.path.exists("model/{}/{}.npz".format(name, fw))