#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:29:04 2018

@author: aguiroz
"""

import numpy as np

from abc import abstractmethod
from interface import NNInterface, NNScreenInterface

from util import check_path
from util import check_model

#screen
from tkinter import Tk, Label, Button, Entry, Frame
from tkinter.ttk import Progressbar

#Plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

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
    
    @classmethod
    def load_weight(cls):
        if cls.model_exist:
            cls.w1 = np.load("model/{}/w1".format(cls.model_name))
            cls.b1 = np.load("model/{}/b1".format(cls.mmodel_name))
            cls.w2 = np.load("model/{}/w2".format(cls.model_name))
            cls.b2 = np.load("model/{}/b2".format(cls.model_name))
        
        return
    
    @classmethod
    def save_weight(cls):
        np.save("w1", cls.w1)
        np.save("b1", cls.b1)
        np.save("w2", cls.w2)
        np.save("b2", cls.b2)
        
        return
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self):
        raise NotImplementedError
    
    def update_progress(self, screen, epoch):
        screen.update_progress(epoch)
        return
    
    def update_plot(self, screen, train, test):
        screen.update_plot(train, test)
        return
    
class NNScreenAbstract(NNScreenInterface, Tk):
    
    @abstractmethod
    def __init__(self, title):
        Tk.__init__(self)
        self.title(title)
        self.geometry("900x800+100+100")
        self.create_model()
        self.set_position()
        
        
        return
    
    def create_model(self):
        #Labels    
        self.lb1 = Label(self, text="Dataset`s Information: ")
        self.lb2 = Label(self, text="Ammount of Data: ")
        self.lb3 = Label(self, text="Ammount of Train Data: ")
        self.lb4 = Label(self, text="Dataset`s Size: ")
        self.lb5 = Label(self, text="Ammount of Data to Test: ")
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
        self.ed1 = Entry(self,)
        self.ed2 = Entry(self,)
        self.ed3 = Entry(self,)
        self.ed4 = Entry(self,)
        self.ed5 = Entry(self,)
        self.ed6 = Entry(self,)
        self.ed7 = Entry(self,)
        self.ed8 = Entry(self,)
        self.ed9 = Entry(self,)
        self.ed10 = Entry(self,)
        self.ed11 = Entry(self,)
        self.ed12 = Entry(self,)
        self.ed13 = Entry(self,)
        
        #Progress
        self.progress = Progressbar(self, orient='horizontal', length=600, mode='determinate')
        
        #Plotting
        self.plot_frame = Frame(self, width=800, height=800, background='white')
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.graph = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        
        #Buttons
        self.btn1 = Button(self, text="-")
        self.btn2 = Button(self, text="+")
        self.btn3 = Button(self, text="-")
        self.btn4 = Button(self, text="+")
        self.btn5 = Button(self, text="Train", command=self.fit)
        self.btn6 = Button(self, text="Predict")
        
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
    
    #@abstractmethod
    def set_info(self):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    def set_maximum_progress(self, value):
        self.progress["maximum"] = value
        return
        
    def update_progress(self, value):
        self.progress["value"] += value
        return

    def update_plot(self, train, test):
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(train, color='orange')
        self.ax.plot(test, color='blue')
        self.graph.draw()
        return