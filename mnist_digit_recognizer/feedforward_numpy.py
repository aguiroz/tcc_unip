#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:28:27 2018

@author: aguiroz
"""

import numpy as np
import matplotlib.pyplot as plt
from util import loadData, getIndicator

def relu(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def classificationRate(target, prediction):
    n_total = 0
    n_correct = 0
    for i in range(len(target)):
        n_total += 1
        if target[i] == prediction[i]:
            n_correct += 1
    return n_correct / n_total

def softmax(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=1, keepdims=True)

def forward(x, w1, b1, w2, b2):
    hidden = relu(x.dot(w1) + b1)
    output = softmax(hidden.dot(w2) + b2)
    return hidden, output

def derivative_w2(hidden, target, output):
    return hidden.T.dot(output - target)

def derivative_w1(x, hidden, output, target, w2):
    diff = (output - target).dot(w2.T) * (hidden * (1 - hidden))
    return (x.T.dot(diff))

def derivative_b2(target, output):
    return (output - target).sum(axis=0)

def derivative_b1(target, output, w2, hidden):
    return ((output - target).dot(w2.T) * hidden * (1 - hidden)).sum()

def getPrediction(output):
    return np.argmax(output, axis=1)

def getError(target, prediction):
    return (np.mean(prediction != target))

def getLoss(target, output):
    tot = target * np.log(output)
    return tot.sum()

def main():
    np.set_printoptions(threshold=np.nan)
    
    x, y = loadData()
    
    x_train = (x[:-1000,] / 255)
    y_train = y[:-1000,]
    yTrain_ind = getIndicator(y_train)
    
    x_test = (x[-1000:] / 255)
    y_test = y[-1000:]
    yTest_ind = getIndicator(y_test)

    #hyperparam
    lr = 0.0005
    epochs = 100
    
    n_examples = x_train.shape[0]
    batch_sz = 300
    n_batch = n_examples // batch_sz
    
    input_sz = 784
    hidden_sz = 500
    output_sz = 10
    ##
    
    w1 = np.random.randn(input_sz, hidden_sz)
    b1 = np.random.randn(hidden_sz)
    w2 = np.random.randn(hidden_sz, output_sz)
    b2 = np.random.randn(output_sz)
    
    lossess = []
    
    for i in range(epochs):
        for j in range(n_batch):
            x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
            y_batch = yTrain_ind[j * batch_sz:(j * batch_sz + batch_sz),]
            
            hidden, output = forward(x_batch, w1, b1, w2, b2)
            
            w2 -= lr * derivative_w2(hidden, y_batch, output)
            b2 -= lr * derivative_b2(y_batch, output)
            w1 -= lr * derivative_w1(x_batch, hidden, output, y_batch, w2)
            b1 -= lr * derivative_b1(y_batch, output, w2, hidden)
        
            if j % 10 == 0:
                hidden_test, output_test = forward(x_test, w1, b1, w2, b2)
                prediction = getPrediction(output_test)
                classification = classificationRate(y_test, prediction)
                loss = -(getLoss(yTest_ind, output_test))
                print('Epoch: {}, Loss: {}, Classification: {}'.format(i, loss, classification * 100))
                lossess.append(loss)
    
    plt.plot(lossess)
    plt.show()
        
        
if __name__ == "__main__":
    main()