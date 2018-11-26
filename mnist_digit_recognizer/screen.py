#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:40:57 2018

@author: aguiroz
"""

from abstract import NNScreenAbstract
from nn import TFMLP, TFCNN, TFRNN
from threading import Thread
from interface import ScreenInterface
from util import save_prediction, load_model_data

#Plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
from tensorflow.train import AdadeltaOptimizer, AdagradDAOptimizer ,AdagradOptimizer, AdamOptimizer, FtrlOptimizer, GradientDescentOptimizer, ProximalAdagradOptimizer, ProximalGradientDescentOptimizer, RMSPropOptimizer

from tkinter import Tk, Toplevel, Label, Button, Entry, filedialog, StringVar, Frame, OptionMenu
    
class TFMLPScreen(NNScreenAbstract):
    
    def __init__(self, features, title='Tensorflow - MLP', train=None, test=None):
        NNScreenAbstract.__init__(self, title, train=train, test=test)
        self.features = features
        self.nn = TFMLP(self)
        
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get()), self.features.lr, self.features.decay, self.features.momentum, self.features.epoch, self.features.test_period, self.features.batch_sz, self.features.optimizer]).start()
        return

    def predict(self):        
        prediction = self.nn.predict(self.test_data)
        data = np.loadtxt(self.test_data.name, dtype=np.uint8, skiprows=1, delimiter=',')
        x = np.array([i.reshape(28, 28) for i in data])
        
        for i in range(x.shape[0]):
            save_prediction(x[i], prediction[i], i, self.nn.model_name)
        return

        
class TFCNNScreen(NNScreenAbstract):
    
    def __init__(self, features, title="Tensorflow - CNN", train=None, test=None):
        NNScreenAbstract.__init__(self, title, train=train, test=test)
        self.features = features
        self.nn = TFCNN(self, batch_sz=self.features.batch_sz)
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get()), self.features.lr, self.features.decay, self.features.momentum, self.features.epoch, self.features.test_period, self.features.batch_sz, self.features.optimizer]).start()
        return
    
    def predict(self):
        prediction = self.nn.predict(self.test_data)
        data = np.loadtxt(self.test_data.name, dtype=np.uint8, skiprows=1, delimiter=',')
        x = np.array([i.reshape(28, 28) for i in data])
        
        for i in range(x.shape[0]):
            save_prediction(x[i], prediction[i], i, self.nn.model_name)
        return

class TFRNNScreen(NNScreenAbstract):
    
    def __init__(self, features, title="Tensorflow - RNN", train=None, test=None):
        NNScreenAbstract.__init__(self, title, train, test)
        self.features = features
        self.nn = TFRNN(self)
        return

    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get()), self.features.lr, self.features.decay, self.features.momentum, self.features.epoch, self.features.test_period, self.features.batch_sz, self.features.optimizer]).start()
        return
    
    def predict(self):
        prediction = self.nn.predict(self.test_data)
        data = np.loadtxt(self.test_data.name, dtype=np.uint8, skiprows=1, delimiter=',')
        x = np.array([i.reshape(28, 28) for i in data])
        
        for i in range(x.shape[0]):
            save_prediction(x[i], prediction[i], i, self.nn.model_name)
        return

    
class MainScreen(ScreenInterface, Tk):
    def __init__(self, title="Home"):
        Tk.__init__(self)
        self.title(title)
        
        self.create_model()
        self.set_position()
        self.title("Home")
        self.geometry("750x200+100+100")
        
        self.features = FeatureScreen(show=True)
        print(self.features.optimizer)
        
        return
    
    def load_mlp(self):
        objMlp = TFMLPScreen(self.features, train=self.loadData.train_data, test=self.loadData.test_data)
        
        return
    
    def load_cnn(self):
        objCnn = TFCNNScreen(self.features, train=self.loadData.train_data, test=self.loadData.test_data)
        
        return
    
    def load_rnn(self):
        objRnn = TFRNNScreen(self.features, train=self.loadData.train_data, test=self.loadData.test_data)

        
        return
        
    def get_features(self):
        self.features = FeatureScreen()
        return
    
    def create_model(self):
        self.lb0 = Label(self, text="Seja Bem-Vindo! :) \n Esse é o nosso Software")
        self.lb1 = Label(self, text="Dataset: ")
        self.lb2 = Label(self, text="Feedforward: ")
        self.lb3 = Label(self, text="CNN: ")
        self.lb4 = Label(self, text="RNN: ")
        self.lb5 = Label(self, text="Parâmetros: ")
        self.lb6 = Label(self, text="Estatísticas: ")
        self.lb7 = Label(self, text="Explicação")
       
        self.btn1 = Button(self, text="Carregar", command=self.load_dataset)
        self.btn2 = Button(self, text="Carregar", command=self.load_mlp)
        self.btn3 = Button(self, text="Carregar", command=self.load_cnn)
        self.btn4 = Button(self, text="Carregar", command=self.load_rnn
                           )
        self.btn5 = Button(self, text="Carregar", command=self.get_features)
        self.btn6 = Button(self, text="Carregar", command=ReportScreen)
        self.btn7 = Button(self, text="Sair", command=self.destroy)
        
        return
    
    def set_position(self):
        self.lb0.grid(row=5, column=0)
        self.lb1.grid(row=0, column=0)
        self.lb2.grid(row=0, column=4)
        self.lb3.grid(row=0, column=6)
        self.lb4.grid(row=1, column=0)
        self.lb5.grid(row=1, column=4)
        self.lb6.grid(row=1, column=6)
        
        self.btn1.grid(row=0, column=2)
        self.btn2.grid(row=0, column=5)
        self.btn3.grid(row=0, column=8)
        self.btn4.grid(row=1, column=2)
        self.btn5.grid(row=1, column=5)
        self.btn6.grid(row=1, column=8)
        self.btn7.grid(row=16, column=16)

        
        return
    
    def load_dataset(self):
        self.loadData = LoadData(self)
        
        return
    
######
    
class LoadData(ScreenInterface, Toplevel):
    
    def __init__(self, root, title="Load Dataset"):        
        Toplevel.__init__(self, root)
        self.title(title)
        
        self.create_model()
        self.set_position()

        self.geometry("300x120+100+100")
    
    def set_position(self):
        
        self.lb1.grid(row=0, column=0)
        self.lb2.grid(row=2, column=0)
        self.ed1.grid(row=0, column=1)
        self.ed2.grid(row=2, column=1)
        self.btn1.grid(row=0, column=2)
        self.btn2.grid(row=2, column=2)
        self.btn3.grid(row=5, column=1)
        self.btn4.grid(row=7, column=1)
        
        return
    
    def add_action(self):
        pass
    
    def set_data(self):
        self.destroy()
        return
    
    def load_train_data(self):
        self.train_data = filedialog.askopenfile(initialdir="./data", title="Select File", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
        self.train_var.set(self.train_data.name)
        return
    
    def load_test_data(self):
        self.test_data = filedialog.askopenfile(initialdir="./data", title="Select File", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
        self.test_var.set(self.test_data.name)
        return
    
    def create_model(self):
        self.lb1 = Label(self, text="Train: ")
        self.lb2 = Label(self, text="Test: ")
    
        self.train_var = StringVar()
        self.test_var = StringVar()
    
        self.ed1 = Entry(self, textvariable=self.train_var)
        self.ed2 = Entry(self, textvariable=self.test_var)
    
        self.btn1 = Button(self, text="Search...", command=self.load_train_data)
        self.btn2 = Button(self, text="Search...", command=self.load_test_data)
        self.btn3 = Button(self, text="Select", command=self.set_data)
        self.btn4 = Button(self, text="Cancel", command=self.destroy)
    
        return
    
class ReportScreen(ScreenInterface, Toplevel):
    def __init__(self, title="Reports"):
        Toplevel.__init__(self)
        
        self.create_model()
        self.set_position()        
        
        self.title(title)
        self.geometry("900x700+100+100")

        self.load_data()
        self.set_data()

        return
    
    def load_data(self):
        
        self.mlp = load_model_data('MLP', 'train_data')
        self.cnn = load_model_data('CNN', 'train_data')
        self.rnn = load_model_data('RNN', 'train_data')
        
        return
    
    def set_data(self):
        
        self.mlp_cost.set(self.mlp[-1, 2][0])
        self.mlp_rate.set("{} %".format(self.mlp[-1, 3][0] / self.mlp[0, -1][0] * 100))
        self.mlp_correct.set("{} / {}".format(self.mlp[-1, 3][0], self.mlp[0, -1][0]))
        self.mlp_time.set(self.mlp[-1, 1][0])
        
        self.cnn_cost.set(self.cnn[-1, 2][0])
        self.cnn_rate.set("{} %".format(self.cnn[-1, 3][0] / self.cnn[0, -1][0] * 100))
        self.cnn_correct.set("{} / {}".format(self.cnn[-1, 3][0], self.cnn[0, -1][0]))
        self.cnn_time.set(self.cnn[-1, 1][0])
        
        self.rnn_cost.set(self.rnn[-1, 2][0])
        self.rnn_rate.set("{} %".format(self.rnn[-1, 3][0] / self.rnn[0, -1][0] * 100))
        self.rnn_correct.set("{} / {}".format(self.rnn[-1, 3][0], self.rnn[0, -1][0]))
        self.rnn_time.set(self.rnn[-1, 1][0])
        
        return
    
    def create_model(self):
        
        self.lb0 = Label(self, text="Estatísticas ")
        self.lb1 = Label(self, text="Feedforward ")
        self.lb2 = Label(self, text="Custo Final: ")
        self.lb3 = Label(self, text="Taxa de Acerto: ")
        self.lb4 = Label(self, text="Qtde de Acerto: ")
        self.lb5 = Label(self, text="Tempo Decorrido: ")
        self.lb6 = Label(self, text="CNN ")
        self.lb7 = Label(self, text="Custo Final: ")
        self.lb8 = Label(self, text="Taxa de Acerto: ")
        self.lb9 = Label(self, text="Qtde de Acerto: ")
        self.lb10 = Label(self, text="Tempo Decorrido: ") 
        self.lb11 = Label(self, text="RNN ")
        self.lb12 = Label(self, text="Custo Final: ")
        self.lb13 = Label(self, text="Taxa de Acerto: ")
        self.lb14 = Label(self, text="Qtde de Acerto: ")
        self.lb15 = Label(self, text="Tempo Decorrido: ")
        self.lb16 = Label(self, text=" ")
        self.lb17 = Label(self, text=" ")
        self.lb18 = Label(self, text=" ")
        self.lb19 = Label(self, text="Gráficos ")
        self.lb20 = Label(self, text="Gráfico Tempo x Iteração: ")
        self.lb21 = Label(self, text="Gráfico Custo x Acerto: ")
        self.lb22 = Label(self, text="Gráfico Custo x Iteração: ")
        
        self.mlp_cost = StringVar()
        self.mlp_rate = StringVar()
        self.mlp_correct = StringVar()
        self.mlp_time = StringVar()
        
        self.cnn_cost = StringVar()
        self.cnn_rate = StringVar()
        self.cnn_correct = StringVar()
        self.cnn_time = StringVar()
        
        self.rnn_cost = StringVar()
        self.rnn_rate = StringVar()
        self.rnn_correct = StringVar()
        self.rnn_time = StringVar()
        
        self.ed1 = Entry(self, textvariable=self.mlp_cost)
        self.ed2 = Entry(self, textvariable=self.mlp_rate)
        self.ed3 = Entry(self, textvariable=self.mlp_correct)
        self.ed4 = Entry(self, textvariable=self.mlp_time)
        self.ed5 = Entry(self, textvariable=self.cnn_cost)
        self.ed6 = Entry(self, textvariable=self.cnn_rate)
        self.ed7 = Entry(self, textvariable=self.cnn_correct)
        self.ed8 = Entry(self, textvariable=self.cnn_time)
        self.ed9 = Entry(self, textvariable=self.rnn_cost)
        self.ed10 = Entry(self, textvariable=self.rnn_rate)
        self.ed11 = Entry(self, textvariable=self.rnn_correct)
        self.ed12 = Entry(self, textvariable=self.rnn_time)
        
        self.btn1 = Button(self, text="Gerar", command=self.plot_train_x_iteration)
        self.btn2 = Button(self, text="Gerar", command=self.plot_cost_x_correct)
        self.btn3 = Button(self, text="Gerar", command=self.plot_cost_x_iteration)
        
        #Plotting
        self.plot_frame = Frame(self, width=100, height=100, background='white')
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.graph = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        
        return
       
    def plot_train_x_iteration(self):
        
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(self.mlp[:, 1], self.mlp[:, 0], color='red')
        self.ax.plot(self.cnn[:, 1], self.mlp[:, 0], color='green')
        self.ax.plot(self.rnn[:, 1], self.mlp[:, 0], color='blue')
        self.graph.draw()
        
        return
    
    def plot_cost_x_correct(self):
        
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(self.mlp[:, 2], self.mlp[:, 3], color='red')
        self.ax.plot(self.cnn[:, 2], self.mlp[:, 3], color='green')
        self.ax.plot(self.rnn[:, 2], self.mlp[:, 3], color='blue')
        self.graph.draw()
        
        return
    
    def plot_cost_x_iteration(self):
        
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(self.mlp[:, 2], self.mlp[:, 0], color='red')
        self.ax.plot(self.cnn[:, 2], self.mlp[:, 0], color='green')
        self.ax.plot(self.rnn[:, 2], self.mlp[:, 0], color='blue')
        self.graph.draw()
        
        return
    
    
        
    
    def set_position(self):
        # Feedforward    
        self.lb1.grid(row=4, column=0)
        self.lb2.grid(row=6, column=0)
        self.lb3.grid(row=7, column=0)
        self.lb4.grid(row=8, column=0)
        self.lb5.grid(row=9, column=0)
    
        # RNN
        self.lb6.grid(row=4, column=2)
        self.lb7.grid(row=6, column=2)
        self.lb8.grid(row=7, column=2)
        self.lb9.grid(row=8, column=2)
        self.lb10.grid(row=9, column=2)
     
        # CNN
        self.lb11.grid(row=4, column=4)
        self.lb12.grid(row=6, column=4)
        self.lb13.grid(row=7, column=4)
        self.lb14.grid(row=8, column=4)
        self.lb15.grid(row=9, column=4)
        
        # Label's de divisão
        self.lb17.grid(row=12, column=0)
        self.lb18.grid(row=15, column=0)
        
        self.ed1.grid(row=6, column=1)
        self.ed2.grid(row=7, column=1)
        self.ed3.grid(row=8, column=1)
        self.ed4.grid(row=9, column=1)
        self.ed5.grid(row=6, column=3)
        self.ed6.grid(row=7, column=3)
        self.ed7.grid(row=8, column=3)
        self.ed8.grid(row=9, column=3)
        self.ed9.grid(row=6, column=5)
        self.ed10.grid(row=7, column=5)
        self.ed11.grid(row=8, column=5)
        self.ed12.grid(row=9, column=5)
        
        # Label's de divisão
        self.lb19.grid(row=16, column=2)
        self.lb20.grid(row=17, column=2)
        self.btn1.grid(row=17, column=3)
        self.lb21.grid(row=18, column=2)
        self.btn2.grid(row=18, column=3)
        self.lb22.grid(row=19, column=2)
        self.btn3.grid(row=19, column=3)

        #Plotting
        self.plot_frame.grid(row=20, column=1, columnspan=8)
        self.ax.grid()
        self.graph.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        return
    
class FeatureScreen(ScreenInterface, Toplevel):
    
    def __init__(self, title='Features', show=False):
        Toplevel.__init__(self)
        
        self.create_model()
        self.set_position()        
        
        self.title(title)
        self.geometry("300x200+100+100")
        
        if show:
            self.set_param()
        
        return
    
    def create_model(self):
        
        #Labels
        self.lb0 = Label(self, text="Learning Rate ")
        self.lb1 = Label(self, text="Decay ")
        self.lb2 = Label(self, text="Momentum ")
        self.lb3 = Label(self, text="Epoch ")
        self.lb4 = Label(self, text="Test Period ")
        self.lb5 = Label(self, text="Batch Size ")
        self.lb6 = Label(self, text="Algorithm ")
        
        #Textvar
        self.lr_var = StringVar(self, value='0.001')
        self.decay_var = StringVar(self, value='0.9')
        self.mom_var = StringVar(self, value='0.0')
        self.epoch_var = StringVar(self, value='10')
        self.test_period_var = StringVar(self, value='10')
        self.batch_sz_var = StringVar(self, value='500')
        self.algorithm_var = StringVar(self, value='RMSPropOptimizer')
        
        #Options
        self.options = {'AdadeltaOptimizer', 'AdagradDAOptimizer', 'AdagradOptimizer', 'AdamOptimizer', 'FtrlOptimizer', 'GradientDescentOptimizer', 'ProximalAdagradOptimizer', 'ProximalGradientDescentOptimizer', 'RMSPropOptimizer'}
        
        #Algorithm
        self.algorithm = {
                'AdadeltaOptimizer': AdadeltaOptimizer, 
                'AdagradDAOptimizer': AdagradDAOptimizer, 
                'AdagradOptimizer': AdagradOptimizer, 
                'AdamOptimizer': AdamOptimizer, 
                'FtrlOptimizer': FtrlOptimizer, 
                'GradientDescentOptimizer': GradientDescentOptimizer, 
                'ProximalAdagradOptimizer': ProximalAdagradOptimizer, 
                'ProximalGradientDescentOptimizer': ProximalGradientDescentOptimizer, 
                'RMSPropOptimizer': RMSPropOptimizer
            }
        
        #Entry
        self.ed0 = Entry(self, textvariable=self.lr_var)
        self.ed1 = Entry(self, textvariable=self.decay_var)
        self.ed2 = Entry(self, textvariable=self.mom_var)
        self.ed3 = Entry(self, textvariable=self.epoch_var)
        self.ed4 = Entry(self, textvariable=self.test_period_var)
        self.ed5 = Entry(self, textvariable=self.batch_sz_var)
        
        #Dropdown
        self.opt = OptionMenu(self, self.algorithm_var, *self.options)
        
        #Button
        self.btnSave = Button(self, text='Save', command= lambda: self.set_param(self.lr_var.get(), self.decay_var.get(), self.mom_var.get(), self.epoch_var.get(), self.test_period_var.get(), self.batch_sz_var.get(), self.algorithm_var.get()))
        self.btnCancel = Button(self, text='Cancel', command=self.cancel)
        
        return
    
    def set_position(self):
        
        self.lb0.grid(row=1, column=0)
        self.lb1.grid(row=2, column=0)
        self.lb2.grid(row=3, column=0)
        self.lb3.grid(row=4, column=0)
        self.lb4.grid(row=5, column=0)
        self.lb5.grid(row=6, column=0)
        self.lb6.grid(row=7, column=0)
        
        self.ed0.grid(row=1, column=1)
        self.ed1.grid(row=2, column=1)
        self.ed2.grid(row=3, column=1)
        self.ed3.grid(row=4, column=1)
        self.ed4.grid(row=5, column=1)
        self.ed5.grid(row=6, column=1)
        
        self.opt.grid(row=7, column=1)
        
        self.btnSave.grid(row=8, column=0)
        self.btnCancel.grid(row=8, column=1)
        
        return
    
    def set_param(self, lr=0.001, decay=0.9, momentum=0.0, epoch=10, test_period=10, batch_sz=500, optimizer='RMSPropOptimizer'):
        
        self.lr = lr 
        self.decay = decay 
        self.momentum = momentum
        self.epoch = epoch
        self.test_period = test_period
        self.batch_sz = batch_sz
        self.optimizer = self.algorithm[optimizer]
        
        self.destroy()
        
        return
    
    def cancel(self):
        
        self.lr = 0.001
        self.decay = 0.9
        self.momentum = 0.0
        self.epoch = 10
        self.test_period = 10
        self.batch_sz = 300
        self.optimizer = self.algorithm['RMSPropOptimizer']
        
        self.destroy()
        
        return
    
    
if __name__ == "__main__":
   
    obj = MainScreen()
    obj.mainloop()
   
