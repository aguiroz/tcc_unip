# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def error_rate(prediction, target):
    return (np.mean(prediction != target))

def getIndicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def classification_rate(target, prediction):
    n_correct = 0
    n_total = 0
    for i in range(len(target)):
        n_total += 1
        if target[i] == prediction[i]:
            n_correct += 1
    return n_correct / n_total

def main():
    data = np.loadtxt(open('data/train.csv'), delimiter=',', skiprows=1)

    x_train = np.array([i[1:] for i in data[:-1000] / 255], dtype=np.float32)
    y_train = np.array([i[0] for i in data[:-1000]], dtype=np.uint8)
    yTrain_ind = getIndicator(y_train)
    
    x_test = np.array([i[1:] for i in data[-1000:] / 255],dtype=np.float32)
    y_test = np.array([i[0] for i in data[-1000:]], dtype=np.uint8)
    yTest_ind = getIndicator(y_test)
    
    #hiperparams
    epochs = 100
    lr = 0.00005
    reg = 0.01

    input_size = 784
    hidden_sz = 300
    output_sz = 10

    n = x_train.shape[0] 
    batch_sz = 200
    n_batch = n // batch_sz
    ##
    
    thX = T.matrix('X')
    thT = T.matrix('T')

    w1 = theano.shared(np.random.randn(input_size, hidden_sz), 'w1')
    b1 = theano.shared(np.random.randn(hidden_sz), 'b1')
    w2 = theano.shared(np.random.randn(hidden_sz, output_sz), 'w2')
    b2 = theano.shared(np.random.randn(output_sz), 'b2')
    
    thZ = T.nnet.relu(thX.dot(w1) + b1)
    thY = T.nnet.softmax(thZ.dot(w2) + b2)
    
    loss = -(thT * T.log(thY)).sum() + reg * ((w1**2).sum() + (b1**2).sum() + (w2**2).sum() + (b2**2).sum())
    prediction = T.argmax(thY, axis=1)
    
    update_w1 = w1 - lr * T.grad(loss, w1)
    update_b1 = b1 - lr * T.grad(loss, b1)
    update_w2 = w2 - lr * T.grad(loss, w2)
    update_b2 = b2 - lr * T.grad(loss, b2)
    
    train = theano.function(inputs=[thX, thT],
                            updates=[(w1, update_w1), (b1, update_b1), (w2, update_w2), (b2, update_b2)],
                            )
    
    get_prediction = theano.function(inputs=[thX, thT],
                                     outputs=[loss, prediction],
                                     )
    
    losses = []
    for i in range(epochs):
        for j  in range(n_batch):
            x_batch = x_train[j * batch_sz:(j * batch_sz + batch_sz),]
            y_batch = yTrain_ind[j * batch_sz:(j * batch_sz + batch_sz),]
            train(x_batch, y_batch)
            
            if j % 20 == 0:
                loss_value, prediction_value = get_prediction(x_test, yTest_ind)
                error = error_rate(prediction_value, y_test)
                c = classification_rate(y_test, prediction_value)
                losses.append(loss_value)
                print("Loss: {}, epoch: {}, error: {}, classification_rate: {}".format(loss_value, i, error * 100, c * 100))
    plt.plot(losses)
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
