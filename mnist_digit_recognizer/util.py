#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:49:52 2018

@author: aguiroz
"""

import numpy as np
import os
from PIL import Image

def load_train_data():
    
    mnist = np.loadtxt("data/train.csv", delimiter=",", skiprows=1, dtype=np.float32)
    
    x_train = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[:2000]])
    y_train = np.array([i[0] for i in mnist[:2000]])

    x_test = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[10000:10500]])
    y_test = np.array([i[0] for i in mnist[10000:10500]])
    
    return x_train, y_train, x_test, y_test
    

def load_data():
    
    train_data = open('data/train.csv')
    test_data = open("data/test.csv")
    
    return train_data, test_data
    
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
    
def save_prediction(data, prediction, index, fw):
    path = "./model/{}/results".format(fw)
    img = Image.fromarray(data.reshape(28,28), 'L')
    img.save('{}/{}/{}.png'.format(path, int(prediction), index))
    return

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
        
def check_model_data(model, file, ext='npy'):
    return os.path.exists("model/{}/{}.{}".format(model, file, ext))

def load_model_data(model, file, ext='npy'):
    return np.load("model/{}/{}.{}".format(model, file, ext))

def save_model_data(data, model, file):
    np.save("model/{}/{}".format(model, file), data)
    return