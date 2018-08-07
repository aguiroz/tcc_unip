#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 7 00:03:15 2018
@author: fabiosoaresv
"""

import matplotlib.pyplot as plt
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding("utf8")

class Rnn:
    label = ["Feedforward", "RNN", "CNN"]

    no_movies = [
        10,
        20,
        3
    ]

    index = np.arange(len(label))
    plt.bar(index, no_movies)
    plt.xlabel("Iteração", fontsize=11)
    plt.ylabel("Custo", fontsize=11)
    plt.xticks(index, label, fontsize=11, rotation=20)
    plt.title("Gráfico Custo x Iteração")
    plt.show()

