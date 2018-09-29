#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:17:22 2018

@author: aguiroz
"""

import numpy as np

#theano
import theano
import theano.tensor as T

from util import load_train_data
from util import get_indicator
from util import sigmoid, softmax
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
    
    def forward(self, x):
        hidden = 1 / (1 + np.exp(-x.dot(self.w1) + self.b1))
        A = hidden.dot(self.w2) + self.b2
        expA = np.exp(A)
        output = expA / expA.sum(axis=1, keepdims=True)
        
        return hidden, output
    
    def derivative_w2(self, hidden, target, output):
        return hidden.T.dot(output - target)
    
    def derivative_b2(self, output, target):
        return (output - target).sum(axis=0)
    
    def derivative_w1(self, x, hidden, output, target):
        diff = (output - target).dot(self.w2.T) * (hidden * (1 - hidden))
        return (x.T.dot(diff))
    
    def derivative_b1(self, target, output, hidden):
        return ((output - target).dot(self.w2.T) * hidden * (1 - hidden)).sum()
        
    
    def create_model(self):
        self.w1 = np.random.randn(self.input_sz, self.hidden_sz).astype(np.float32)
        self.b1 = np.random.randn(self.hidden_sz).astype(np.float32)
        self.w2 = np.random.randn(self.hidden_sz, self.output_sz).astype(np.float32)
        self.b2 = np.random.randn(self.output_sz).astype(np.float32)
        
        return
    
    def getLoss(self, target, output):
        #print("target: {}".format(target))
        #print("output: {}".format(output))
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
        return np.mean(prediction != target)
    
    
    def fit(self, screen, learning_rate=0.000001, batch_sz=200):
        np.set_printoptions(threshold=np.nan)
        x_train, y_train, x_test, y_test = load_train_data()
        n_batches = x_train.shape[0] // self.batch_sz
        
        y_train_ind = get_indicator(y_train)
        yTest_ind = get_indicator(y_test)
        
        self.train_loss = []
        self.test_loss = []
        
        for i in range(self.epoch):
            for j in range(n_batches):    
                
                x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
                y_batch = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz),]
                Y_b = y_train[j * batch_sz:(j * batch_sz + batch_sz),]
                
                hidden, output = self.forward(x_batch)
                
                self.w2 -= learning_rate * self.derivative_w2(hidden, y_batch, output)
                self.b2 -= learning_rate * self.derivative_b2(y_batch, output)
                self.w1 -= learning_rate * self.derivative_w1(x_batch, hidden, output, y_batch)
                self.b1 -= learning_rate * self.derivative_b1(y_batch, output, hidden)
                
                p_train = self.predict(output)
                tr_loss = self.getLoss(Y_b, p_train)
                
                hidden_test, output_test = self.forward(x_test)
                prediction = self.predict(output_test)
                classification = self.classificationRate(y_test, prediction)
                ts_loss = self.getLoss(yTest_ind, output_test)
                error = self.getError(y_test, prediction)

                print('Epoch: {}, Loss: {}, Error: {} Classification: {}'.format(i, tr_loss, error * 100, classification * 100))
                    
                self.train_loss.append(tr_loss)
                self.test_loss.append(ts_loss) 
                
                self.update_progress(screen, i)
                self.update_plot(screen, self.train_loss, self.test_loss)

        return
    
    def predict(self, output):
        return np.argmax(output, axis=1)
    
##################################################
    
class TMLP(NNAbstract):
    
    def __init__(self, model_name='MLP', fw='theano', input_sz=784, hidden_sz=300, output_sz=10):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        
        self.create_model()
        
        return
    
    def error_rate(self, prediction, target):
        return (np.mean(prediction != target))
    
    def create_model(self, lr=0.0005, reg=0.01):
        self.thX = T.matrix('X')
        self.thT = T.matrix('T')
    
        self.w1 = theano.shared(np.random.randn(self.input_sz, self.hidden_sz), 'w1')
        self.b1 = theano.shared(np.random.randn(self.hidden_sz), 'b1')
        self.w2 = theano.shared(np.random.randn(self.hidden_sz, self.output_sz), 'w2')
        self.b2 = theano.shared(np.random.randn(self.output_sz), 'b2')
        
        self.thZ = T.nnet.relu(self.thX.dot(self.w1) + self.b1)
        self.thY = T.nnet.softmax(self.thZ.dot(self.w2) + self.b2)
    
        self.loss = -(self.thT * T.log(self.thY)).sum() + reg * ((self.w1**2).sum() + (self.b1**2).sum() + (self.w2**2).sum() + (self.b2**2).sum())
        
        self.prediction = T.argmax(self.thY, axis=1)
        
        update_w1 = self.w1 - lr * T.grad(self.loss, self.w1)
        update_b1 = self.b1 - lr * T.grad(self.loss, self.b1)
        update_w2 = self.w2 - lr * T.grad(self.loss, self.w2)
        update_b2 = self.b2 - lr * T.grad(self.loss, self.b2)
        
        self.train = theano.function(inputs=[self.thX, self.thT],
                                updates=[(self.w1, update_w1), 
                                         (self.b1, update_b1), 
                                         (self.w2, update_w2), 
                                         (self.b2, update_b2)],
                                )
        
        self.get_prediction = theano.function(inputs=[self.thX, self.thT],
                                         outputs=[self.loss, self.prediction],
                                         )

        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, elapsed=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, test_cost=test_cost, test_error=test_error, test_correct=test_correct, iteration=epoch, batch=batch)        
        return
    
    def predict(self):
        pass
    
    def fit(self, screen, epoch=10, batch_sz=300, test_period=10):
        
        x_train, y_train, x_test, y_test = load_train_data()
        n_batch = x_train.shape[0] // batch_sz
        y_train_ind = get_indicator(y_train)
        
        screen.set_maximum_progress(epoch * n_batch)

        self.train_losses = []
        self.test_losses = []
        
        for i in range(epoch):
            for j in range(n_batch):
                
                x_test, y_test = shuffle(x_test, y_test)
                y_test_ind = get_indicator(y_test)
                
                x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
                y_batch = y_train[j * batch_sz:(j * batch_sz + batch_sz)]
                y_batch_ind = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz),]
                
                self.train(x_batch, y_batch_ind)
                
                train_loss, prediction = self.get_prediction(x_batch, y_batch_ind)
                train_error = self.error_rate(prediction, y_batch)
                
                if j % test_period == 0:
                    test_loss, test_prediction = self.get_prediction(x_test[:500,], y_test_ind[:500,])
                    test_error = self.error_rate(test_prediction, y_test[:500])
                    
                    print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, test_loss, test_error * 100))
                
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                
                screen.update_plot(self.train_losses, self.test_losses)
                screen.update_progress()
                
                self.update_info(screen, train_loss, train_error * 100, 0, test_loss, test_error * 100, 0, i, j)
                
                print("Epoch: {}, Loss: {}, Error: {}".format(i, train_loss, train_error * 100))
                
        
        return
    
if __name__ == '__main__':
    obj = TMLP()
    obj.fit()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
