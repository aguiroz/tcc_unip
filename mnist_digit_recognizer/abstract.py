#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:29:04 2018

@author: aguiroz
"""

import numpy as np

from abc import abstractmethod
from interface import NNInterface, NNScreenInterface

#util
from util import check_path
from util import check_model
from time import time

#screen
from tkinter import Label, Button, Entry, Frame, StringVar, Toplevel
from tkinter.ttk import Progressbar

#Plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class NNAbstract(NNInterface):
    
    @abstractmethod
    def __init__(self, model_name, fw):
        check_path(model_name)
        self.model_name = model_name
        self.model_exist = check_model(model_name, fw)
        return
    
    @abstractmethod
    def create_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def update_info(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_weight(self):
        raise NotImplementedError
    
    @abstractmethod
    def save_weight(self):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_prediction(self):
        raise NotImplementedError
        
    @abstractmethod
    def predict(self):
        raise NotImplementedError
        
    def error_rate(self, prediction, target):
        return (np.mean(prediction != target))
    
    def update_progress(self, screen, epoch):
        screen.update_progress(epoch)
        return
    
    def update_plot(self, screen, train, test):
        screen.update_plot(train, test)
        return
    
class NNScreenAbstract(NNScreenInterface, Toplevel):
    
    @abstractmethod
    def __init__(self, title, train=None, test=None):
        Toplevel.__init__(self)
        self.title(title)
        self.geometry("900x800+100+100")
        self.create_model()
        self.set_position()

        if train is not None:
            self.train_data = train
            self.get_dataset_size(train, 'train')
        if test is not None:
            self.test_data = test
            self.get_dataset_size(test, 'test')

        self.set_info()

        
        return
    
    def get_dataset_size(self, data, dataset):
        if dataset == 'train':
            self.ammount_var.set(sum(1 for line in data) - 1)
        else:
            self.dataset_size_var.set(sum(1 for line in data) - 1)
            
        return
    
    def set_info(self, ammount=0, dataset_size=0, qtd_train=0, qtd_test=0, train_cost=0, train_error=0, train_correct=0, test_cost=0, test_error=0, test_correct=0, iteration=0, batch=0, start=0):
        
        if ammount != 0 or self.ammount_var.get() == "":
            self.ammount_var.set(ammount)
        
        if dataset_size != 0 or self.dataset_size_var.get() == "":
            self.dataset_size_var.set(dataset_size)
        
        if qtd_train != 0 or self.qtd_train_var.get() == "":
            self.qtd_train_var.set(qtd_train)
        
        if qtd_test != 0 or self.qtd_test_var.get() == "":
            self.qtd_test_var.set(qtd_test)
        
        if train_cost != 0 or self.train_cost_var.get() == "":
            self.train_cost_var.set(train_cost)
        
        if train_error != 0 or self.train_error_var.get() == "":
            self.train_error_var.set(train_error)
        
        if train_correct != 0 or self.train_correct_var.get() == "":
            self.train_correct_var.set(train_correct)
        
        if test_cost != 0 or self.test_cost_var.get() == "":
            self.test_cost_var.set(test_cost)
        
        if test_error != 0 or self.test_error_var.get() == "":
            self.test_error_var.set(test_error)
        
        if test_correct != 0 or self.test_correct_var.get() == "":
            self.test_correct_var.set(test_correct)
        
        if iteration != 0 or self.iteration_var.get() == "":
            self.iteration_var.set(iteration)
        
        if batch != 0 or self.batch_var.get() == "":
            self.batch_var.set(batch)
        
        if start != 0:
            self.elapsed_var.set(time() - start)
        else:
            self.elapsed_var.set(0)
        
        return
    
    #qtd_train_var
    #qtd_test_var
    
    def increase_train(self):
        self.qtd_train_var.set(int(self.qtd_train_var.get()) + 200)
        return
    
    def decrease_train(self):
        self.qtd_train_var.set(int(self.qtd_train_var.get()) - 200)
        return
    
    def increase_test(self):
        self.qtd_test_var.set(int(self.qtd_test_var.get()) + 200)
        return
    
    def decrease_test(self):
        self.qtd_test_var.set(int(self.qtd_test_var.get()) - 200)
        return
    
    
    def create_model(self):
        
        #string vars
        self.ammount_var = StringVar()
        self.dataset_size_var = StringVar()
        self.qtd_train_var = StringVar()
        self.qtd_test_var = StringVar()
        self.train_cost_var = StringVar()
        self.train_error_var = StringVar()
        self.train_correct_var = StringVar()
        
        self.test_cost_var = StringVar()
        self.test_error_var = StringVar()
        self.test_correct_var = StringVar()      
        
        self.iteration_var = StringVar()
        self.batch_var = StringVar()
        self.elapsed_var = StringVar()
        
        #Labels    
        self.lb1 = Label(self, text="Dataset`s Information: ")
        self.lb2 = Label(self, text="Train Dataset`s Size: ")
        self.lb3 = Label(self, text="Ammount of Train Data: ")
        self.lb4 = Label(self, text="Predict Dataset`s Size: ")
        self.lb5 = Label(self, text="Ammount of Test Data: ")
        self.lb6 = Label(self, text="Train ")
        self.lb7 = Label(self, text="Cost: ")
        self.lb8 = Label(self, text="Error: ")
        self.lb9 = Label(self, text="Correct: ")
        self.lb10 = Label(self, text="Iteration: ")
        self.lb11 = Label(self, text="Batch: ")
        self.lb12 = Label(self, text="Test ")
        self.lb13 = Label(self, text="Cost: ")
        self.lb14 = Label(self, text="Error: ")
        self.lb15 = Label(self, text="Correct: ")
        self.lb16 = Label(self, text="Elapsed Time: ")
        self.lb17 = Label(self, text="Progress: ")
        self.lb18 = Label(self, text="Train x Test: ")
        self.lb19 = Label(self, text=" ")
        self.lb20 = Label(self, text=" ")
        self.lb21 = Label(self, text=" ")
        self.lb22 = Label(self, text=" ")
        self.lb23 = Label(self, text=" ")
        
        #Entries
        self.ed1 = Entry(self, textvariable=self.ammount_var) # ammount_of_data
        self.ed2 = Entry(self, textvariable=self.dataset_size_var) # dataset_size
        self.ed3 = Entry(self, textvariable=self.qtd_train_var) # qtd_train
        self.ed4 = Entry(self, textvariable=self.qtd_test_var) # qtd_test
        self.ed5 = Entry(self, textvariable=self.train_cost_var) # train_cost
        self.ed6 = Entry(self, textvariable=self.train_error_var) # train_error
        self.ed7 = Entry(self, textvariable=self.train_correct_var) # train_correct
        self.ed8 = Entry(self, textvariable=self.iteration_var) # iteration
        self.ed9 = Entry(self, textvariable=self.batch_var) # batch 
        self.ed10 = Entry(self, textvariable=self.test_cost_var) # test_cost 
        self.ed11 = Entry(self, textvariable=self.test_error_var) # test_error
        self.ed12 = Entry(self, textvariable=self.test_correct_var) # test_correct
        self.ed13 = Entry(self, textvariable=self.elapsed_var) # elapsed_time
        
        #Progress
        self.progress = Progressbar(self, orient='horizontal', length=600, mode='determinate')
        
        #Plotting
        self.plot_frame = Frame(self, width=800, height=800, background='white')
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.graph = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        
        #Buttons
        self.btn1 = Button(self, text="-", command=self.decrease_train)
        self.btn2 = Button(self, text="+", command=self.increase_train)
        self.btn3 = Button(self, text="-", command=self.decrease_test)
        self.btn4 = Button(self, text="+", command=self.increase_test)
        self.btn5 = Button(self, text="Train", command=self.fit)
        self.btn6 = Button(self, text="Predict", command=self.predict)
        
        return
    
    def set_position(self):
        
        #Labels
        self.lb1.grid(row=0, column=0)
        self.lb2.grid(row=1, column=0)
        self.lb3.grid(row=1, column=2)
        self.lb4.grid(row=2, column=0)
        self.lb5.grid(row=2, column=2)
        
        #Separator
        self.lb19.grid(row=3, column=0)
        self.lb20.grid(row=4, column=0)
        
        self.lb6.grid(row=5, column=0)
        self.lb7.grid(row=6, column=0)
        self.lb8.grid(row=7, column=0)
        self.lb9.grid(row=8, column=0)
        self.lb10.grid(row=9, column=0)
        self.lb11.grid(row=10, column=0)
        self.lb12.grid(row=5, column=2)
        self.lb13.grid(row=6, column=2)
        self.lb14.grid(row=7, column=2)
        self.lb15.grid(row=8, column=2)
        self.lb16.grid(row=9, column=2)
        self.lb17.grid(row=16, column=0)
        
        #Separator
        self.lb21.grid(row=12, column=0)
        self.lb22.grid(row=15, column=0)
        self.lb23.grid(row=19, column=0)
        
        self.ed1.grid(row=1, column=1)
        self.ed2.grid(row=2, column=1)
        self.ed3.grid(row=1, column=4)
        self.ed4.grid(row=2, column=4)
        self.ed5.grid(row=6, column=1)
        self.ed6.grid(row=7, column=1)
        self.ed7.grid(row=8, column=1)
        self.ed8.grid(row=9, column=1)
        self.ed9.grid(row=10, column=1)
        self.ed10.grid(row=6, column=3)
        self.ed11.grid(row=7, column=3)
        self.ed12.grid(row=8, column=3)
        self.ed13.grid(row=9, column=3)
        
        #Progress
        self.progress.grid(row=16, column=1, columnspan=4)
        
        #Plotting
        self.plot_frame.grid(row=20, column=1, columnspan=8)
        self.ax.grid()
        self.graph.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        self.btn1.grid(row=1, column=3)
        self.btn2.grid(row=1, column=5)
        self.btn3.grid(row=2, column=3)
        self.btn4.grid(row=2, column=5)
        self.btn5.grid(row=14, column=1)
        self.btn6.grid(row=14, column=3)

        return
    
    @abstractmethod    
    def predict(self):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    def set_maximum_progress(self, value):
        self.progress["maximum"] = value
        return
        
    def update_progress(self, value=1):
        self.progress["value"] += value
        return

    def update_plot(self, train, test):
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(train, color='orange')
        self.ax.plot(test, color='blue')
        self.graph.draw()
        return
    



