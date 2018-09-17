#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:29:04 2018

@author: aguiroz
"""

import numpy as np
from tkinter import Tk
from abc import abstractmethod
from interface import NNInterface
from util import check_path
from util import check_model


class NNAbstract(NNInterface):
    
    @abstractmethod
    def __init__(self, model_name, fw, screen: Tk):
        check_path(model_name)
        self.model_name = model_name
        self.model_exist = check_model(model_name, fw)
        self.screen = screen
        return
    
    @abstractmethod
    def create_model(self):
        pass
    
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
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    def update_plot(self, epoch):
        self.screen.update_progress(epoch)
        return
    
    def update_progress(self):
        pass
    
