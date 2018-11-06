#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:40:57 2018

@author: aguiroz
"""

from abstract import NNScreenAbstract
from nn import MLP, TMLP, TFMLP, TFCNN, TRNN

from threading import Thread

class NMLPScreen(NNScreenAbstract):
    
    def __init__(self, title="Numpy - MLP"):
        NNScreenAbstract.__init__(self, title)
        self.progress["maximum"] = 9
        
        self.nn = MLP()
        
        return
        
    def fit(self):
        Thread(target=self.nn.fit, args=[self]).start()
        return
    
    def predict(self):
        pass
    
        
class TMLPScreen(NNScreenAbstract):
    
    def __init__(self, title="Theano - MLP"):
        NNScreenAbstract.__init__(self, title)
        
        self.nn = TMLP()
        
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self]).start()
        return
    
    def predict(self):
        pass
    
    
class TFMLPScreen(NNScreenAbstract):
    
    def __init__(self, title='Tensorflow - MLP'):
        NNScreenAbstract.__init__(self, title)
        
        self.nn = TFMLP()
        
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self]).start()
        return
    
    def predict(self):
        pass
    
class TCNNScreen(NNScreenAbstract):
    
    def __init__(self, title="Theano - CNN"):
        NNScreenAbstract.__init__(self, title)
        
        return
    
    def fit(self):
        pass
    
    def predict(self):
        pass
        
class TFCNNScreen(NNScreenAbstract):
    
    def __init__(self, title="Tensorflow - CNN"):
        NNScreenAbstract.__init__(self, title)
        self.nn = TFCNN()
        return
    
    def fit(self):
        Thread(target=self.nn.fit, args=[self]).start()
        return
    
    def predict(self):
        pass
    
class TRNNScreen(NNScreenAbstract):
    
    def __init__(self, title="Tensorflow - RNN"):
        NNScreenAbstract.__init__(self, title)
        self.nn = TRNN()
        return

    def fit(self):
        Thread(target=self.nn.fit, args=[self]).start()
        return
    
    def predict(self):
        pass

class LoadData(ScreenInterface, Tk):
    
    def __init__(self, title="Load Dataset"):        
        Tk.__init__(self)
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
    
    def create_model(self):
        self.lb1 = Label(self, text="Train: ")
        self.lb2 = Label(self, text="Test: ")
    
        self.ed1 = Entry(self,)
        self.ed2 = Entry(self,)
    
        self.btn1 = Button(self, text="Search...")
        self.btn2 = Button(self, text="Search...")
        self.btn3 = Button(self, text="Save")
        self.btn4 = Button(self, text="Cancel")
    
        return    
    
if __name__ == "__main__":
    obj = TFMLPScreen()
    obj.mainloop()
   
