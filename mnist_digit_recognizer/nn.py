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

#tf
import tensorflow as tf

from util import get_indicator, classificationRate, load_train_data
from time import time
from sklearn.utils import shuffle
from abstract import NNAbstract

# CNN
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

#RNN
from tensorflow.contrib import rnn

        
class TFMLP(NNAbstract):
    def __init__(self, model_name='MLP', fw='theano', input_sz=784, hidden_sz=300, output_sz=10):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
                
        self.create_model()
        
        return
    
    def create_model(self, lr=0.0005, decay=0.99, momentum=0.9):
        self.tfX = tf.placeholder(tf.float32, shape=(None, self.input_sz), name='x')
        self.tfT = tf.placeholder(tf.float32, shape=(None, self.output_sz), name='t')
        
        self.w1 = tf.Variable(np.random.randn(self.input_sz, self.hidden_sz), dtype=np.float32)
        self.b1 = tf.Variable(np.random.randn(self.hidden_sz), dtype=np.float32)
        self.w2 = tf.Variable(np.random.randn(self.hidden_sz, self.output_sz), dtype=np.float32)
        self.b2 = tf.Variable(np.random.randn(self.output_sz), dtype=np.float32)
        
        self.tfZ = tf.nn.relu(tf.matmul(self.tfX, self.w1) + self.b1)
        self.tfY = tf.matmul(self.tfZ, self.w2) + self.b2
        
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.tfY, labels=self.tfT))
        self.train = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=momentum).minimize(self.loss)
        self.predict = tf.argmax(self.tfY, 1)
        self.init = tf.global_variables_initializer()
        
        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, start=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, 
                        test_cost=test_cost, test_error=test_error, test_correct=test_correct, 
                        iteration=epoch, batch=batch, start=start)        
        return
    
    def split_data(self, data, qtd_train, qtd_test):
        mnist = np.loadtxt(data.name, delimiter=',', skiprows=1, dtype=np.float32)
        x_train = np.array([i[1:] / 255 for i in mnist[:qtd_train]])
        y_train = np.array([i[0] for i in mnist[:qtd_train]])
        
        x_test = np.array([i[1:] / 255 for i in mnist[qtd_train:qtd_train + qtd_test]])
        y_test = np.array([i[0]  for i in mnist[qtd_train:qtd_train + qtd_test]])
        return x_train, y_train, x_test, y_test

    def fit(self, screen, train_data, qtd_train, qtd_test, test_period=10, epoch=10, batch_sz=100):
        print("*"*500, qtd_train, qtd_test)
        self.create_model()
        
        #x_train, y_train, x_test, y_test = load_train_data()
        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        
        
        n_batch = qtd_train // batch_sz
        
        screen.set_maximum_progress(epoch * n_batch)

        self.train_losses = []
        self.test_losses = []
        
        start = time()
        
        with tf.Session() as session:
            session.run(self.init)
            for i in range(epoch):
                x_train, y_train = shuffle(x_train, y_train)
                y_train_ind = get_indicator(y_train)
                
                x_test, y_test = shuffle(x_test, y_test)
                y_test_ind = get_indicator(y_test)
                
                for j in range(n_batch):
                    x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz)]
                    y_batch = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz)]
                    
                    session.run(self.train, feed_dict={self.tfX: x_batch, self.tfT: y_batch})
                    train_loss = session.run(self.loss, feed_dict={self.tfX: x_train, self.tfT: y_train_ind})
                    prediction = session.run(self.predict, feed_dict={self.tfX: x_train})
                    train_error = self.error_rate(prediction, y_train)
                    
                    if j % test_period == 0:
                        test_loss = session.run(self.loss, feed_dict={self.tfX: x_test, self.tfT: y_test_ind})
                        test_prediction = session.run(self.predict, feed_dict={self.tfX: x_test})
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                        print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, test_loss, test_error * 100))
                
                    train_qtd_correct = classificationRate(y_train, prediction)
                    
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                     
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                 train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                 test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                 i, j, start)
                    
        return
    
    def predict(self):
        pass
    
    
##################################################

class TFCNN(NNAbstract):
    def __init__(self, model_name='CNN', fw='tensorflow', input_sz=784, hidden_sz=1000, output_sz=10, batch_sz=500):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.batch_sz = batch_sz
                
        self.create_model()
        
        return
    
    def convpool(self, X, W, b):
        conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, b)
        pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(pool_out)

    
    def init_filter(self, shape, poolsz):
        w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
        return w.astype(np.float32)
    
    def create_model(self, poolsz=(2, 2), lr=0.0001, decay=0.99, momentum=0.9):
        
        W1_shape = (5, 5, 1, 20)
        W1_init = self.init_filter(W1_shape, poolsz)
        b1_init = np.zeros(W1_shape[-1], dtype=np.float32)
        
        W2_shape = (5, 5, 20, 50)
        W2_init = self.init_filter(W2_shape, poolsz)
        b2_init = np.zeros(W2_shape[-1], dtype=np.float32)
        
        W3_init = np.random.randn(2450, self.hidden_sz) / np.sqrt(W2_shape[-1]*8*8 + self.hidden_sz)
        b3_init = np.zeros(self.hidden_sz, dtype=np.float32)
        W4_init = np.random.randn(self.hidden_sz, self.output_sz) / np.sqrt(self.hidden_sz + self.output_sz)
        b4_init = np.zeros(self.output_sz, dtype=np.float32)
        
        self.X = tf.placeholder(tf.float32, shape=(self.batch_sz, 28, 28, 1), name='X')
        self.T = tf.placeholder(tf.int32, shape=(self.batch_sz,), name='T')
        
        self.W1 = tf.Variable(W1_init.astype(np.float32))
        self.b1 = tf.Variable(b1_init.astype(np.float32))
        self.W2 = tf.Variable(W2_init.astype(np.float32))
        self.b2 = tf.Variable(b2_init.astype(np.float32))
        self.W3 = tf.Variable(W3_init.astype(np.float32))
        self.b3 = tf.Variable(b3_init.astype(np.float32))
        self.W4 = tf.Variable(W4_init.astype(np.float32))
        self.b4 = tf.Variable(b4_init.astype(np.float32))
        
        self.Z1 = self.convpool(self.X, self.W1, self.b1)
        self.Z2 = self.convpool(self.Z1, self.W2, self.b2)
        self.Z2_shape = self.Z2.get_shape().as_list()
        self.Z2r = tf.reshape(self.Z2, [self.Z2_shape[0], np.prod(self.Z2_shape[1:])])
        self.Z3 = tf.nn.relu(tf.matmul(self.Z2r, self.W3) + self.b3)
        self.Yish = tf.matmul(self.Z3, self.W4) + self.b4
        
        self.cost = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.Yish,
                labels=self.T
            )
        )
            
        self.train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(self.cost)
        self.predict_op = tf.argmax(self.Yish, 1)

        self.init = tf.initialize_all_variables()
        
        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, start=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, test_cost=test_cost, test_error=test_error, test_correct=test_correct, iteration=epoch, batch=batch, start=start)
        return
    
    def split_data(self, train_data, qtd_train, qtd_test):
        mnist = np.loadtxt(train_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        
        x_train = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[:qtd_train]])
        y_train = np.array([i[0] for i in mnist[:qtd_train]])
        x_test = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[qtd_train:qtd_train + qtd_test]])
        y_test = np.array(i[0] for i in mnist[qtd_train:qtd_train + qtd_test])
        
        return x_train, y_train, x_test, y_test

    def fit(self, screen, train_data, qtd_train, qtd_test, epoch=25, test_period=10, batch_sz=500):
        
        self.create_model()
        
        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        
        n_batch = qtd_train // batch_sz
        
        screen.set_maximum_progress(epoch * n_batch)

        self.train_losses = []
        self.test_losses = []
        
        start = time()
        
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(epoch):
                
                for j in range(n_batch):
                    x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
                    y_batch = y_train[j * batch_sz:(j * batch_sz + batch_sz),]
                    
                    sess.run(self.train_op, feed_dict={self.X: x_batch, self.T: y_batch})
                    
                    prediction = np.zeros(qtd_train)
                    train_loss = 0
                    
                    for k in range(n_batch):
                        train_loss += sess.run(self.cost, feed_dict={self.X: x_train[k * batch_sz:(k * batch_sz + batch_sz),], self.T: y_train[k * batch_sz:(k * batch_sz + batch_sz)]})
                        prediction[k * batch_sz:(k * batch_sz + batch_sz)] = sess.run(self.predict_op, feed_dict={self.X: x_train[k * batch_sz:(k * batch_sz + batch_sz)]})
                    
                    train_error = self.error_rate(prediction, y_train)
                    train_qtd_correct = classificationRate(y_train, prediction)
                    
                    if j % test_period == 0:
                        test_loss = 0
                        test_prediction = np.zeros(qtd_test)
                        
                        for k in range(qtd_test // batch_sz):
                            test_loss += sess.run(self.cost, feed_dict={self.X: x_test[k * batch_sz:(k * batch_sz + batch_sz)], self.T: y_test[k * batch_sz:(k * batch_sz + batch_sz)]})
                            test_prediction[k * batch_sz:(k * batch_sz + batch_sz)] = sess.run(self.predict_op, feed_dict={self.X: x_test[k * batch_sz:(k * batch_sz + batch_sz)]})
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                            
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                    
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                     train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                     test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                     i, j, start)
                
                    
                            
                    print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, test_loss, train_error * 100))
        return
    
    def predict(self):
        pass
    
##################################################
    
class TFRNN(NNAbstract):
    def __init__(self, model_name='RNN', fw='tensorflow', input_sz=28, timesteps=28, hidden_sz=128, output_sz=10, epoch=300):
        NNAbstract.__init__(self, model_name, fw)

        self.input_sz = input_sz
        self.timesteps = timesteps
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz

        self.epoch = epoch

        self.create_model()

        return

    def RNN(self, x, weights, biases):

        x = tf.unstack(x, self.timesteps, 1)
        lstm_cell = rnn.BasicLSTMCell(self.hidden_sz, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def create_model(self, lr=0.001):


        self.X = tf.placeholder("float", [None, self.timesteps, self.input_sz])
        self.Y = tf.placeholder("float", [None, self.output_sz])

        self.weights = {
		    'out': tf.Variable(tf.random_normal([self.hidden_sz, self.output_sz]))
		}

        self.biases = {
		    'out': tf.Variable(tf.random_normal([self.output_sz]))
		}


        self.logits = self.RNN(self.X, self.weights, self.biases)
        self.prediction = tf.nn.softmax(self.logits)
        self.predict_op = tf.argmax(self.prediction, 1)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		    logits=self.logits, labels=self.Y))
        
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.init = tf.global_variables_initializer()


        return
    
    def split_data(self, train_data, qtd_train, qtd_test):
        mnist = np.loadtxt(train_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        
        x_train = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[:qtd_train]])
        y_train = np.array([i[0] for i in mnist[:qtd_train]])
        x_test = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[qtd_train:qtd_train + qtd_test]])
        y_test = np.array([i[0] for i in mnist[qtd_train:qtd_train + qtd_test]])
        
        return x_train, y_train, x_test, y_test

    def fit(self, screen, train_data, qtd_train, qtd_test, epoch=300, batch_sz=50, test_period=10):

        self.create_model()
        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        n_batch = qtd_train // batch_sz

        y_train_ind = get_indicator(y_train)
        y_test_ind = get_indicator(y_test)

        screen.set_maximum_progress(epoch * n_batch)
        
        start = time()

        with tf.Session() as sess:

            sess.run(self.init)

            self.train_losses = []
            self.test_losses = []

            for i in range(self.epoch):
                for j in range(n_batch):
                    x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz)]
                    y_batch = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz)]
    
                    sess.run(self.train_op, feed_dict={self.X: x_batch, self.Y: y_batch})
                    
                    train_loss = sess.run(self.loss_op, feed_dict={self.X: x_train, self.Y: y_train_ind})
                    prediction = sess.run(self.predict_op, feed_dict={self.X: x_train})
                    train_error = self.error_rate(prediction, y_train)
                    
                    
                    if j % test_period == 0:
                        test_loss = sess.run(self.loss_op, feed_dict={self.X: x_test, self.Y: y_test_ind})
                        test_prediction = sess.run(self.predict_op, feed_dict={self.X: x_test})
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                        
                    train_qtd_correct = classificationRate(y_train, prediction)
                
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                    
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                     train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                     test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                     i, j, start)
                    
                    print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, train_loss, train_error * 100))
		


        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, start=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, test_cost=test_cost, test_error=test_error, test_correct=test_correct, iteration=epoch, batch=batch, start=start)        
        return

    def predict(self):

        return
    
if __name__ == "__main__":
    obj = TFCNN()
    obj.fit()



