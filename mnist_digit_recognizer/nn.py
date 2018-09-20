#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:17:22 2018

@author: aguiroz
"""

import numpy as np
from util import load_train_data
from util import get_indicator
from util import relu, softmax
from sklearn.utils import shuffle
from abstract import NNAbstract


class MLP(NNAbstract):
    
    def __init__(self, model_name='MLP', fw='numpy', input_sz=784, hidden_sz=300, output_sz=10, epoch=10, batch_sz=10):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.epoch = epoch
        self.batch_sz = batch_sz
        
        self.create_model()
        return
    
    def forward(self, x, w1, b1, w2, b2):
        hidden = relu(x.dot(w1) + b1)
        output = softmax(hidden.dot(w2) + b2)
        
        return hidden, output
    
    def derivative_w2(self, hidden, target, output):
        return hidden.T.dot(output - target)
    
    def derivative_b2(self, output, target):
        return (output - target).sum(axis=0)
    
    def derivative_w1(self, x, hidden, output, target, w2):
        diff = (output - target).dot(w2.T) * (hidden * (1 - hidden))
        return x.T.dot(diff)
    
    def derivative_b1(self, target, output, w2, hidden):
        return ((output - target).dot(w2.T) * hidden * (1 - hidden)).sum()
        
    
    def create_model(self):
        self.w1 = np.random.randn(self.input_sz, self.hidden_sz)
        self.b1 = np.random.randn(self.hidden_sz)
        self.w2 = np.random.randn(self.hidden_sz, self.output_sz)
        self.b2 = np.random.randn(self.output_sz)
        
        return
    
    def getLoss(self, target, output):
        tot = target * np.log(output)
        return tot.sum()
    
    def classificationRate(self, target, prediction):
        n_total = 0
        n_correct = 0
        for i in range(len(target)):
            n_total += 1
            if target[i] == prediction[i]:
                n_correct += 1
        return n_correct / n_total
    
    def getError(self, target, prediction):
        return (np.mean(prediction != target))
    
    
    def fit(self, screen, learning_rate=1e-5, batch_sz=500):
        x_train, y_train, x_test, y_test = load_train_data()
        n_batches = x_train.shape[0] // self.batch_sz

        
        yTest_ind = get_indicator(y_test)
        
        self.loss = []
        
        for i in range(self.epoch):
            for j in range(n_batches):
                x_train, y_train = shuffle(x_train, y_train)
                
                y_train_ind = get_indicator(y_train)
                
                x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz)]
                y_batch = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz)]
                
                hidden, output = self.forward(x_batch, self.w1, self.b1, self.w2, self.b2)
                
                self.w2 -= learning_rate * self.derivative_w2(hidden, y_batch, output)
                self.b2 -= learning_rate * self.derivative_b2(y_batch, output)
                self.w1 -= learning_rate * self.derivative_w1(x_batch, hidden, output, y_batch, self.w2)
                self.b1 -= learning_rate * self.derivative_b1(y_batch, output, self.w2, hidden)
                
                self.update_progress(screen, i)
                
                if j % 10 == 0:
                    hidden_test, output_test = self.forward(x_test, self.w1, self.b1, self.w2, self.b2)
                    prediction = self.predict(output_test)
                    classification = self.classificationRate(y_test, prediction)
                    loss = -(self.getLoss(yTest_ind, output_test))
                    error = self.getError(y_test, prediction)
                
                    print('Epoch: {}, Loss: {}, Error: {} Classification: {}'.format(i, loss, error * 100, classification * 100))
                
        return
    
    def predict(self, output):
        return np.argmax(output, axis=1)

from tkinter import Tk

if __name__ == "__main__":
    screen = Tk()
    obj = MLP(screen)
    obj.fit()
    
    