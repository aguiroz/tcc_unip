#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:46:53 2018

@author: aguiroz
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1)))
    A = Z.dot(W2)
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def derivative_w2(Z, T, Y):
    N, K = T.shape
    ret4 = Z.T.dot(T - Y)
    return ret4


def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)


    return ret2


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def save(data, index, result):
    img = Image.fromarray(data.reshape(28,28), 'L')
    img.save('{}/{}.png'.format(result, index))
    return


def main():
    
    mnist = np.loadtxt(open('data/train.csv'), delimiter=',', skiprows=1, dtype=np.uint8)

    D = 784 
    M = 550
    K = 10 

    X = np.vstack([i[1:] for i in mnist])    
    Y = np.array([i[0] for i in mnist])
    
    N = len(Y)

    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    W1 = np.random.randn(D, M)
    W2 = np.random.randn(M, K)

    learning_rate = 1e-5
    costs = []
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2)
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = classification_rate(Y, P)
        print('iteration {}:'.format(epoch))
        print("cost: {} classification_rate: {}".format(c, r * 1000))
        costs.append(c)

        W2 += learning_rate * derivative_w2(hidden, T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)

    plt.plot(costs)
    plt.show()
    
    print('loading test data...')
    test_data = np.loadtxt(open('data/test.csv'), delimiter=',', skiprows=1, dtype=np.uint8)
    print('ok')
    
    test = np.vstack(i for i in test_data)
    
    output, hidden = forward(test, W1, None, W2, None)
    
    result = np.argmax(output, axis=1)

    print('Saving results...')
    
    for i in range(output.shape[0]):
        save(test[i], i, result[i])
        
        
    print('done!')
    
    

if __name__ == '__main__':
    main()
