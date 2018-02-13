#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:46:53 2018

@author: aguiroz
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def forward(X, W1, W2):
    hidden = sigmoid(X.dot(W1))
    output = softmax(hidden.dot(W2))
    return output, hidden


def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def derivative_w2(hidden, target, output):
    N, K = target.shape
    ret4 = hidden.T.dot(target - output)
    return ret4


def derivative_w1(X, hidden, target, output, W2):
    N, D = X.shape
    M, K = W2.shape

    dH = (target - output).dot(W2.T) * hidden * (1 - hidden)
    ret2 = X.T.dot(dH)


    return ret2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def cost(target, output):
    tot = target * np.log(output)
    return tot.sum()


def save(data, index, result):
    img = Image.fromarray(data.reshape(28,28), 'L')
    img.save('{}/{}.png'.format(result, index))
    return


def main():
    print('Loading train data...')
    mnist = np.loadtxt(open('data/train.csv'), delimiter=',', skiprows=1, dtype=np.uint8)

    input_size = 784
    hidden_size = 550
    output_size = 10

    X = np.vstack([i[1:] for i in mnist])
    Y = np.array([i[0] for i in mnist])

    N = len(Y)

    target = np.zeros((N, output_size))
    for i in range(N):
        target[i, Y[i]] = 1

    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)

    learning_rate = 1e-5
    costs = []

    for epoch in range(2000):
        output, hidden = forward(X, W1, W2)
        c = cost(target, output)
        P = np.argmax(output, axis=1)
        r = classification_rate(Y, P)
        print('iteration {}:'.format(epoch))
        print("cost: {} classification_rate: {}".format(c, r * 100))
        costs.append(c)

        W2 += learning_rate * derivative_w2(hidden, target, output)
        W1 += learning_rate * derivative_w1(X, hidden, target, output, W2)

    plt.plot(costs)
    plt.show()

    print('loading test data...')
    test_data = np.loadtxt(open('data/test.csv'), delimiter=',', skiprows=1, dtype=np.uint8)
    print('ok')

    test = np.vstack(i for i in test_data)

    output, hidden = forward(test, W1, W2)

    result = np.argmax(output, axis=1)

    print('Saving results...')

    for i in range(output.shape[0]):
        save(test[i], i, result[i])


    print('done!')



if __name__ == '__main__':
    main()
