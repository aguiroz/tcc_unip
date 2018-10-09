#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:40:57 2018

@author: aguiroz
"""

from abstract import NNScreenAbstract
from nn import MLP, TMLP, TFMLP

from threading import Thread
from multiprocessing import Process
import queue

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
        p = Process(target=self.nn.fit, args=(None,))
        p.start()
        p.join()
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
        
    
    
if __name__ == "__main__":
    obj = TFMLPScreen()
    obj.mainloop()
    