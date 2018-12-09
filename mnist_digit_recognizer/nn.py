#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:17:22 2018

@author: aguiroz
"""

#memory usage
import os
import psutil

import numpy as np

#tf
import tensorflow as tf

from util import get_indicator, classificationRate, check_model_data, load_model_data, save_model_data
from time import time
from sklearn.utils import shuffle
from abstract import NNAbstract

#RNN
from tensorflow.contrib import rnn

        
class TFMLP(NNAbstract):
    def __init__(self, screen, model_name='MLP', fw='tensorflow', input_sz=784, hidden_sz=1000, output_sz=10):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
                
        self.create_model()
        
        if self.load_train_data():
            screen.update_plot(self.train_losses, self.test_losses)
        else:
            self.train_losses = []
            self.test_losses = []
        
        return
    
    def load_weight(self):
        model = self.model_name
        
        if not check_model_data(model, "w1"):
            return False
        if not check_model_data(model, "b1"):
            return False
        if not check_model_data(model, "w2"):
            return False
        if not check_model_data(model, "b2"):
            return False
        
        w1 = tf.Variable(load_model_data(model, "w1"))
        b1 = tf.Variable(load_model_data(model, "b1"))
        w2 = tf.Variable(load_model_data(model, "w2"))
        b2 = tf.Variable(load_model_data(model, "b2"))
        
        self.w1 = tf.Variable(w1)
        self.b1 = tf.Variable(b1)
        self.w2 = tf.Variable(w2)
        self.b2 = tf.Variable(b2)
        
        return True
    
    def save_weight(self, sess):
        model = self.model_name
        save_model_data(self.w1.eval(sess), model, "w1")
        save_model_data(self.b1.eval(sess), model, "b1")
        save_model_data(self.w2.eval(sess), model, "w2")
        save_model_data(self.b2.eval(sess), model, "b2")
        return
    
    def create_model(self, lr=0.0005, decay=0.99, momentum=0.9):
        self.tfX = tf.placeholder(tf.float32, shape=(None, self.input_sz), name='x')
        self.tfT = tf.placeholder(tf.float32, shape=(None, self.output_sz), name='t')
        
        if not self.load_weight():
            self.w1 = tf.Variable(np.random.randn(self.input_sz, self.hidden_sz), dtype=np.float32)
            self.b1 = tf.Variable(np.random.randn(self.hidden_sz), dtype=np.float32)
            self.w2 = tf.Variable(np.random.randn(self.hidden_sz, self.output_sz), dtype=np.float32)
            self.b2 = tf.Variable(np.random.randn(self.output_sz), dtype=np.float32)
        
        self.tfZ = tf.nn.relu(tf.matmul(self.tfX, self.w1) + self.b1)
        self.tfY = tf.matmul(self.tfZ, self.w2) + self.b2
        
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.tfY, labels=self.tfT))
        self.predict_op = tf.argmax(self.tfY, 1)
        self.init = tf.global_variables_initializer()
        
        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, start=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, 
                        test_cost=test_cost, test_error=test_error, test_correct=test_correct, 
                        iteration=epoch, batch=batch, start=start)        
        return
    
    def predict(self, input_data: open):
        data = np.loadtxt(input_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        x = np.array([i for i in data])
        prediction = self.get_prediction(x)
        return prediction
    
    def split_data(self, data, qtd_train, qtd_test):
        mnist = np.loadtxt(data.name, delimiter=',', skiprows=1, dtype=np.float32)
        x_train = np.array([i[1:] / 255 for i in mnist[:qtd_train]])
        y_train = np.array([i[0] for i in mnist[:qtd_train]])
        
        x_test = np.array([i[1:] / 255 for i in mnist[qtd_train:qtd_train + qtd_test]])
        y_test = np.array([i[0]  for i in mnist[qtd_train:qtd_train + qtd_test]])
        return x_train, y_train, x_test, y_test

    def get_prediction(self, x, session=None):
        if session is None:
            self.create_model()
            session = tf.Session()
            session.run(tf.initialize_all_variables())
        
        prediction = session.run(self.predict_op, feed_dict={self.tfX: x})
        
        return prediction

    def fit(self, screen, train_data, qtd_train, qtd_test, lr=0.001, decay=0.9, momentum=0.0, epoch=10, test_period=10, batch_sz=500, optimizer=tf.train.RMSPropOptimizer):
        self.create_model()
        #self.train = optimizer(lr, decay=decay, momentum=momentum).minimize(self.loss)
        self.train = tf.train.AdagradOptimizer(lr).minimize(self.loss)

        process = psutil.Process(os.getpid())
        
        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        
        n_batch = qtd_train // batch_sz
        
        screen.set_maximum_progress(epoch * n_batch)

        start = time()
        self.train_data = []
        
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
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
                    prediction = self.get_prediction(x_train, session)
                    train_error = self.error_rate(prediction, y_train)
                    
                    if j % test_period == 0:
                        test_loss = session.run(self.loss, feed_dict={self.tfX: x_test, self.tfT: y_test_ind})
                        test_prediction = self.get_prediction(x_test, session) 
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                        print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, test_loss, test_error * 100))
                
                    train_qtd_correct = classificationRate(y_train, prediction)
                    
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                    self.train_data.append([[i], [time() - start], [train_loss], [train_qtd_correct], [qtd_train], [process.memory_info().rss]])
                     
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                 train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                 test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                 i, j, start)
            self.save_weight(session)
            self.save_train_data()
                    
        return
    
    
    
##################################################

class TFCNN(NNAbstract):
    def __init__(self, screen, model_name='CNN', fw='tensorflow', input_sz=784, hidden_sz=1000, output_sz=10, batch_sz=500):
        NNAbstract.__init__(self, model_name, fw)
        
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.batch_sz = batch_sz
                
        self.create_model()
        if self.load_train_data():
            screen.update_plot(self.train_losses, self.test_losses)
        else:
            self.train_losses = []
            self.test_losses = []
        
        return
    
    def load_weight(self):
        model = self.model_name
        
        if not check_model_data(model, "w1"):
            return False
        if not check_model_data(model, "b1"):
            return False
        if not check_model_data(model, "w2"):
            return False
        if not check_model_data(model, "b2"):
            return False
        if not check_model_data(model, "w3"):
            return False
        if not check_model_data(model, "b3"):
            return False
        if not check_model_data(model, "w4"):
            return False
        if not check_model_data(model, "b4"):
            return False
        
        w1 = tf.Variable(load_model_data(model, "w1"))
        b1 = tf.Variable(load_model_data(model, "b1"))
        w2 = tf.Variable(load_model_data(model, "w2"))
        b2 = tf.Variable(load_model_data(model, "b2"))
        w3 = tf.Variable(load_model_data(model, "w3"))
        b3 = tf.Variable(load_model_data(model, "b3"))
        w4 = tf.Variable(load_model_data(model, "w4"))
        b4 = tf.Variable(load_model_data(model, "b4"))
        
        self.w1 = tf.Variable(w1)
        self.b1 = tf.Variable(b1)
        self.w2 = tf.Variable(w2)
        self.b2 = tf.Variable(b2)
        self.w3 = tf.Variable(w3)
        self.b3 = tf.Variable(b3)
        self.w4 = tf.Variable(w4)
        self.b4 = tf.Variable(b4)
        
        return True
    
    def save_weight(self, sess):
        
        model = self.model_name
        save_model_data(self.w1.eval(sess), model, "w1")
        save_model_data(self.b1.eval(sess), model, "b1")
        save_model_data(self.w2.eval(sess), model, "w2")
        save_model_data(self.b2.eval(sess), model, "b2")
        save_model_data(self.w3.eval(sess), model, "w3")
        save_model_data(self.b3.eval(sess), model, "b3")
        save_model_data(self.w4.eval(sess), model, "w4")
        save_model_data(self.b4.eval(sess), model, "b4")
        
        return

    
    def convpool(self, x, w, b):
        conv_out = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, b)
        pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(pool_out)

    
    def init_filter(self, shape, poolsz):
        w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
        return w.astype(np.float32)
    
    def create_model(self, poolsz=(2, 2), lr=0.0001, decay=0.99, momentum=0.9):
        
        w1_shape = (5, 5, 1, 20)
        w1_init = self.init_filter(w1_shape, poolsz)
        b1_init = np.zeros(w1_shape[-1], dtype=np.float32)
        
        w2_shape = (5, 5, 20, 50)
        w2_init = self.init_filter(w2_shape, poolsz)
        b2_init = np.zeros(w2_shape[-1], dtype=np.float32)
        
        w3_init = np.random.randn(2450, self.hidden_sz) / np.sqrt(w2_shape[-1]*8*8 + self.hidden_sz)
        b3_init = np.zeros(self.hidden_sz, dtype=np.float32)
        w4_init = np.random.randn(self.hidden_sz, self.output_sz) / np.sqrt(self.hidden_sz + self.output_sz)
        b4_init = np.zeros(self.output_sz, dtype=np.float32)
        
        self.X = tf.placeholder(tf.float32, shape=(self.batch_sz, 28, 28, 1), name='X')
        self.T = tf.placeholder(tf.int32, shape=(self.batch_sz, ), name='T')
        
        if not self.load_weight():
            self.w1 = tf.Variable(w1_init.astype(np.float32))
            self.b1 = tf.Variable(b1_init.astype(np.float32))
            self.w2 = tf.Variable(w2_init.astype(np.float32))
            self.b2 = tf.Variable(b2_init.astype(np.float32))
            self.w3 = tf.Variable(w3_init.astype(np.float32))
            self.b3 = tf.Variable(b3_init.astype(np.float32))
            self.w4 = tf.Variable(w4_init.astype(np.float32))
            self.b4 = tf.Variable(b4_init.astype(np.float32))
        
        self.Z1 = self.convpool(self.X, self.w1, self.b1)
        self.Z2 = self.convpool(self.Z1, self.w2, self.b2)
        self.Z2_shape = self.Z2.get_shape().as_list()
        self.Z2r = tf.reshape(self.Z2, [self.Z2_shape[0], np.prod(self.Z2_shape[1:])])
        self.Z3 = tf.nn.relu(tf.matmul(self.Z2r, self.w3) + self.b3)
        self.Ypred = tf.matmul(self.Z3, self.w4) + self.b4
        
        self.cost = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.Ypred,
                labels=self.T
            )
        )
            
        self.predict_op = tf.argmax(self.Ypred, 1)

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
        y_test = np.array([i[0] for i in mnist[qtd_train:qtd_train + qtd_test]])
        
        print("Train: {} - {}\nTest: {} - {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        
        return x_train, y_train, x_test, y_test
    
    def predict(self, input_data: open):
        data = np.loadtxt(input_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        x = np.array([i.reshape(28, 28) for i in data])
        x = np.expand_dims(x, axis=3)
        
        prediction = self.get_prediction(x)        
            
        return prediction

    def get_prediction(self, x, session=None):
        if session is None:
            self.batch_sz = x.shape[0]
            self.create_model()
            session = tf.Session()
            session.run(self.init)
            
        prediction = session.run(self.predict_op, feed_dict={self.X: x})
        
        return prediction

    def fit(self, screen, train_data, qtd_train, qtd_test, lr=0.001, decay=0.9, momentum=0.0, epoch=10, test_period=10, batch_sz=500, optimizer=tf.train.RMSPropOptimizer):
        
        self.create_model()
        #self.train_op = optimizer(lr, momentum=momentum, decay=decay).minimize(self.cost)
        self.train_op = tf.train.AdagradOptimizer(lr).minimize(self.cost)


        process = psutil.Process(os.getpid())
        
        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        
        n_batch = qtd_train // batch_sz
        
        screen.set_maximum_progress(epoch * n_batch)

        start = time()
        self.train_data = []
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(epoch):
                
                for j in range(n_batch):
                    x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
                    y_batch = y_train[j * batch_sz:(j * batch_sz + batch_sz),]
                    
                    sess.run(self.train_op, feed_dict={self.X: x_batch, self.T: y_batch})
                    
                    prediction = np.zeros(qtd_train)
                    train_loss = 0
                    
                    for k in range(n_batch):
                        train_loss += sess.run(self.cost, feed_dict={self.X: x_train[k * batch_sz:(k * batch_sz + batch_sz),], self.T: y_train[k * batch_sz:(k * batch_sz + batch_sz)]})
                        prediction[k * batch_sz:(k * batch_sz + batch_sz)] = self.get_prediction(x_train[k * batch_sz:(k * batch_sz + batch_sz)], sess) 
                    
                    train_error = self.error_rate(prediction, y_train)
                    train_qtd_correct = classificationRate(y_train, prediction)
                    
                    if j % test_period == 0:
                        test_loss = 0
                        test_prediction = np.zeros(qtd_test)
                        
                        for k in range(qtd_test // batch_sz):
                            test_loss += sess.run(self.cost, feed_dict={self.X: x_test[k * batch_sz:(k * batch_sz + batch_sz),], self.T: y_test[k * batch_sz:(k * batch_sz + batch_sz)]})
                            test_prediction[k * batch_sz:(k * batch_sz + batch_sz)] = self.get_prediction(x_test[k * batch_sz:(k * batch_sz + batch_sz)], sess) 
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                            
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                    self.train_data.append([[i], [time() - start], [train_loss], [train_qtd_correct], [qtd_train], [process.memory_info().rss]])
                    
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                     train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                     test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                     i, j, start)
                
                    
                            
                    print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, test_loss, train_error * 100))
            self.save_weight(sess)
            self.save_train_data()
            
        return
    
    
##################################################
    
class TFRNN(NNAbstract):
    def __init__(self, screen, model_name='RNN', fw='tensorflow', input_sz=28, timesteps=28, hidden_sz=1000, output_sz=10):
        NNAbstract.__init__(self, model_name, fw)

        self.input_sz = input_sz
        self.timesteps = timesteps
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        
        self.model_path = 'model/RNN/model.ckpt'

        self.create_model()
        if self.load_train_data():
            screen.update_plot(self.train_losses, self.test_losses)
        else:
            self.train_losses = []
            self.test_losses = []

        return

    def RNN(self, x, weights, biases):

        x = tf.unstack(x, self.timesteps, 1)
        lstm_cell = rnn.BasicLSTMCell(self.hidden_sz, forget_bias=1.0, reuse=tf.AUTO_REUSE)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    
    def save_weight(self, sess):
        model = self.model_name
        save_model_data(self.weights['out'].eval(sess), model, 'w')
        save_model_data(self.biases['out'].eval(sess), model, "b")
        
        return

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
        
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        return
    
    def load_weight(self, sess):
        if check_model_data('RNN', 'model', ext='ckpt'):
            self.saver.restore(sess, self.model_path)
        pass
    
    def split_data(self, train_data, qtd_train, qtd_test):
        mnist = np.loadtxt(train_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        
        x_train = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[:qtd_train]])
        y_train = np.array([i[0] for i in mnist[:qtd_train]])
        x_test = np.array([i[1:].reshape(28, 28) / 255 for i in mnist[qtd_train:qtd_train + qtd_test]])
        y_test = np.array([i[0] for i in mnist[qtd_train:qtd_train + qtd_test]])
        
        return x_train, y_train, x_test, y_test

    def predict(self, input_data: open):
        data = np.loadtxt(input_data.name, delimiter=',', skiprows=1, dtype=np.float32)
        x = np.array([i.reshape(28, 28) for i in data])
        prediction = self.get_prediction(x)
        
        return prediction
    
    def get_prediction(self, x, sess=None):
        
        if sess is None:
        
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                self.load_weight(sess)
                prediction = sess.run(self.predict_op, feed_dict={self.X: x})
        else:
            prediction = sess.run(self.predict_op, feed_dict={self.X: x})
            
        return prediction
    
    def fit(self, screen, train_data, qtd_train, qtd_test, lr=0.001, decay=0.9, momentum=0.0, epoch=10, test_period=10, batch_sz=500, optimizer=tf.train.GradientDescentOptimizer):
	    
        self.create_model()
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss_op)
        
        process = psutil.Process(os.getpid())

        x_train, y_train, x_test, y_test = self.split_data(train_data, qtd_train, qtd_test)
        n_batch = qtd_train // batch_sz

        y_train_ind = get_indicator(y_train)
        y_test_ind = get_indicator(y_test)

        screen.set_maximum_progress(epoch * n_batch)
        
        start = time()
        self.train_data = []

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())
            self.load_weight(sess)
            for i in range(epoch):
                for j in range(n_batch):
                    x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz)]
                    y_batch = y_train_ind[j * batch_sz:(j * batch_sz + batch_sz)]
    
                    sess.run(self.train_op, feed_dict={self.X: x_batch, self.Y: y_batch})
                    
                    train_loss = sess.run(self.loss_op, feed_dict={self.X: x_train, self.Y: y_train_ind})
                    prediction = self.get_prediction(x_train, sess) 
                    train_error = self.error_rate(prediction, y_train)
                    
                    
                    if j % test_period == 0:
                        test_loss = sess.run(self.loss_op, feed_dict={self.X: x_test, self.Y: y_test_ind})
                        test_prediction = self.get_prediction(x_test, sess) 
                        test_error = self.error_rate(test_prediction, y_test)
                        test_qtd_correct = classificationRate(y_test, test_prediction)
                        
                    train_qtd_correct = classificationRate(y_train, prediction)
                
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
                    self.train_data.append([[i], [time() - start], [train_loss], [train_qtd_correct], [qtd_train], [process.memory_info().rss]])
                    
                    screen.update_plot(self.train_losses, self.test_losses)
                    screen.update_progress()
                    
                    self.update_info(screen, 
                                     train_loss, train_error * 100, '{} / {}'.format(train_qtd_correct, qtd_train),
                                     test_loss, test_error * 100, '{} / {}'.format(test_qtd_correct, qtd_test), 
                                     i, j, start)
                    
                    print("### Test: Epoch: {}, Loss: {}, Error: {}".format(i, train_loss, train_error * 100))
            self.save_weight(sess)
            self.save_train_data()
            self.saver.save(sess, 'model/RNN/model.ckpt')

        return
    
    def update_info(self, screen, train_cost, train_error, tain_correct, test_cost, test_error, test_correct, epoch, batch, start=0):
        screen.set_info(train_cost=train_cost, train_error=train_error, train_correct=tain_correct, test_cost=test_cost, test_error=test_error, test_correct=test_correct, iteration=epoch, batch=batch, start=start)        
        return
