#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 6 23:58:35 2018
@author: fabiosoaresv
"""

import matplotlib.pyplot as plt
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding("utf8")

class Feedforward:
    label = ["Feedforward", "RNN", "CNN"]

    no_movies = [
        10,
        20,
        3
    ]

    index = np.arange(len(label))
    plt.bar(index, no_movies)
    plt.xlabel("Tempo", fontsize=11)
    plt.ylabel("Iteração", fontsize=11)
    plt.xticks(index, label, fontsize=11, rotation=20)
    plt.title("Gráfico Tempo x Iteração")
    plt.show()

