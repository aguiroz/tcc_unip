#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:40:57 2018

@author: aguiroz
"""

from abstract import NNScreenAbstract
from nn import MLP, TMLP

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
        
    
    
if __name__ == "__main__":
    obj = TMLPScreen()
    obj.mainloop()