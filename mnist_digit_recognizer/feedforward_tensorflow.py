#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:44:36 2018

@author: aguiroz
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def getIndicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(prediction, target):
    return np.mean(prediction != target)

def classification_rate(prediction, target):
    n_correct = 0
    n_total = 0
    for i in range(len(target)):
        n_total += 1
        if target[i] == prediction [i]:
            n_correct += 1
    return n_correct / n_total

def main():
    data = np.loadtxt(open("data/train.csv"), delimiter=",", skiprows=1)
    x_train = np.array([i[1:] for i in data[:-1000]], dtype=np.float32)
    y_train = np.array([i[0] for i in data[:-1000]], dtype=np.float32)
    yTrain_ind = getIndicator(y_train)
    
    x_test = np.array([i[1:] for i in data[-1000:]], dtype=np.float32)
    y_test = np.array([i[0] for i in data[-1000:]], dtype=np.float32)
    yTest_ind = getIndicator(y_test)
    
    #hyperparam
    epochs = 100
    lr = 0.00005
    reg = 0.01
    
    input_sz = 784
    hidden_sz = 300
    output_sz = 10
    
    n = x_train.shape[0]
    batch_sz = 300
    n_batch = n // batch_sz
    ##
    
    tfX = tf.placeholder(tf.float32, shape=(None, input_sz), name='x')
    tfT = tf.placeholder(tf.float32, shape=(None, output_sz), name='t')
    
    w1 = tf.Variable(np.random.randn(input_sz, hidden_sz), dtype=np.float32)
    b1 = tf.Variable(np.random.randn(hidden_sz), dtype=np.float32)
    w2 = tf.Variable(np.random.randn(hidden_sz, output_sz), dtype=np.float32)
    b2 = tf.Variable(np.random.randn(output_sz), dtype=np.float32)
    
    tfZ = tf.nn.relu(tf.matmul(tfX, w1) + b1)
    tfY = tf.matmul(tfZ, w2) + b2
    
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tfY, labels=tfT))
    train = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(loss)
    predict = tf.argmax(tfY, 1)
    init = tf.global_variables_initializer()
    
    lossess = []
    
    with tf.Session() as session:
        session.run(init)
        
        for i in range(epochs):
            for j in range(n_batch):
                x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz)]
                y_batch = yTrain_ind[j * batch_sz:(j * batch_sz + batch_sz)]
                
                session.run(train, feed_dict={tfX: x_batch, tfT: y_batch})
                if j % 20 == 0:
                    loss_value = session.run(loss, feed_dict={tfX: x_test, tfT: yTest_ind})
                    prediction = session.run(predict, feed_dict={tfX: x_test})
                    error = error_rate(prediction, y_test)
                    classification = classification_rate(prediction, y_test)
                    print("Loss: {}, epoch: {}, error: {}, classification: {}".format(loss_value, i, error * 100, classification * 100))
                    lossess.append(loss_value)
    plt.plot(lossess)
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    