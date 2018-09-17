# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class NNInterface(metaclass=ABCMeta):
    
    @classmethod
    @abstractmethod
    def create_model(cls):
        raise NotImplementedError
        
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self):
        raise NotImplementedError
        
    @abstractmethod
    def update_plot(self):
        raise NotImplementedError
        
    @abstractmethod
    def update_progress(self):
        raise NotImplementedError
        
    @abstractmethod
    def load_weight(self):
        raise NotImplementedError
        
    @abstractmethod
    def save_weight(self):
        raise NotImplementedError
            