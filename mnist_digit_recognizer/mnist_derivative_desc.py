#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:21:34 2018

@author: aguiroz
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z = sigmoid(X.dot(W1) + b1)
    y = softmax(z.dot(W2) + b2)
    return z, y

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return n_correct / n_total

def grad_w2(hidden, target, output):
    return hidden.T.dot(output - target)

def grad_b2(target, output):
    return (target - output).sum(axis=0)

def grad_w1(X, hidden, target, output, W2):
    return X.T.dot(((output - target).dot(W2.T) * (hidden * (1 - hidden))))

def grad_b1(target, output, W2, hidden):
    return ((target - output).dot(W2.T) * hidden * (1 - hidden)).sum(axis=0)

def cost(target, output, reg, W1, b1, W2, b2):
    tot = (target * np.log(output) + reg * (W1**2).sum() + (b1**2).sum() + (W2**2).sum() + (b2**2).sum()).sum()
    return tot

def save(data, index, result):
    img = Image.fromarray(data.reshape(28,28), 'L')
    img.save('{}/{}.png'.format(result, index))
    return

def main():
    print('Loading train data...')
    mnist = np.loadtxt(open('data/train.csv'), delimiter=',', skiprows=1, dtype=np.uint8)
    
    input_size = 784
    hidden_size = 300
    output_size = 10
    
    X = np.vstack(i[1:] for i in mnist)
    Y = np.vstack(i[0] for i in mnist)
    
    N = len(Y)
    
    target = np.zeros((N, output_size))
    for i in range(N):
        target[i, Y[i]] = 1
        
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.random.randn(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.random.randn(output_size)
    
    learning_rate = 1e-5
    reg = 0.001
    costs = []
    
    for epoch in range(1000):
        hidden, output = forward(X, W1, b1, W2, b2)
        c = cost(target, output, reg, W1, b1, W2, b2)
        P = np.argmax(output, axis=1)
        r = classification_rate(Y, P)
        
        print('iteration: {}'.format(epoch))
        print('cost: {}, classification_rate: {}'.format(c, r * 100))
        
        costs.append(c)
    
        W2 -= learning_rate * grad_w2(hidden, target, output)
        b2 -= learning_rate * grad_b2(target, output)
        W1 -= learning_rate * grad_w1(X, hidden, target, output, W2)
        b1 -= learning_rate * grad_b1(target, output, W2, hidden)
        
    plt.plot(costs)
    plt.show()
    
    print('Loading test data...')
    
    test_data = np.loadtxt(open('data/test.csv'), delimiter=',', skiprows=1, dtype=np.uint8)
    test = np.vstack(i for i in test_data)
    
    hidden, output = forward(test, W1, W2)
    predict = np.argmax(output, axis=1)
    
    print('Saving results...')
    
    for i in range(output.shape[0]):
        save(test[i], i, predict[i])
        
    print('Done!')

if __name__ == '__main__':
    main()    