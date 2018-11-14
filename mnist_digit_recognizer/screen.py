#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:40:57 2018

@author: aguiroz
"""

from abstract import NNScreenAbstract
from nn import TFMLP, TFCNN, TFRNN
from util import load_data
from threading import Thread
from interface import ScreenInterface

from tkinter import Tk, Toplevel, Label, Button, Entry, filedialog, StringVar
    
class TFMLPScreen(NNScreenAbstract):
    
    def __init__(self, title='Tensorflow - MLP', train=None, test=None):
        NNScreenAbstract.__init__(self, title, train=train, test=test)
        self.nn = TFMLP()
        
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get())]).start()
        return
    
    def predict(self):
        pass

        
class TFCNNScreen(NNScreenAbstract):
    
    def __init__(self, title="Tensorflow - CNN", train=None, test=None):
        NNScreenAbstract.__init__(self, title, train, test)
        self.nn = TFCNN()
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get())]).start()
        return
    
    def predict(self):
        pass
    
class TFRNNScreen(NNScreenAbstract):
    
    def __init__(self, title="Tensorflow - RNN", train=None, test=None):
        NNScreenAbstract.__init__(self, title, train, test)
        self.nn = TFRNN()
        return

    def fit(self):
        Thread(target=self.nn.fit, args=[self, self.train_data, int(self.qtd_train_var.get()), int(self.qtd_test_var.get())]).start()
        return
    
    def predict(self):
        pass

    
class MainScreen(ScreenInterface, Tk):
    def __init__(self, title="Home"):
        Tk.__init__(self)
        self.title(title)
        
        self.create_model()
        self.set_position()
        self.title("Home")
        self.geometry("750x200+100+100")
        
        train, test = load_data()
        
        self.train_data = train
        self.test_data = test

        
        return
    
    def load_mlp(self):
        obj = TFMLPScreen(self, train=self.train_data, test=self.test_data)
        
        return
    
    def load_cnn(self):
        obj = TFCNNScreen(train=self.train_data, test=self.test_data)
        
        return
    
    def load_rnn(self):
        obj = TFRNNScreen(train=self.train_data, test=self.test_data)

        
        return
        
    
    def create_model(self):
        self.lb0 = Label(self, text="Seja Bem-Vindo! :) \n Esse é o nosso Software")
        self.lb1 = Label(self, text="Dataset: ")
        self.lb2 = Label(self, text="Feedforward: ")
        self.lb3 = Label(self, text="RNN: ")
        self.lb4 = Label(self, text="CNN: ")
        self.lb5 = Label(self, text="Parâmetros: ")
        self.lb6 = Label(self, text="Estatísticas: ")
        self.lb7 = Label(self, text="Explicação")
       
        self.btn1 = Button(self, text="Carregar", command=self.load_dataset)
        self.btn2 = Button(self, text="Carregar", command=self.load_mlp)
        self.btn3 = Button(self, text="Carregar", command=self.load_cnn)
        self.btn4 = Button(self, text="Carregar", command=self.load_rnn
                           )
        self.btn5 = Button(self, text="Carregar")
        self.btn6 = Button(self, text="Carregar")
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
        self.train_data = filedialog.askopenfile(initialdir=".", title="Select File", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
        self.train_var.set(self.train_data.name)
        return
    
    def load_test_data(self):
        self.test_data = filedialog.askopenfile(initialdir=".", title="Select File", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
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
    
if __name__ == "__main__":
   
    obj = MainScreen()
    obj.mainloop()
   
