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
    
    
if __name__ == "__main__":
    obj = TFCNNScreen()
    obj.mainloop()
   
