#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:17:35 2018
@author: fabiosoaresv
"""
from Tkinter import *

class Index:
    janela = Tk()

    lb1 = Label(janela, text="Carregar Dataset: ")
    
    btn1 = Button(janela, text="Carregar")

    lb1.grid(row=0, column=0)
    btn1.grid(row=0, column=2)

    janela.title("Home")
    janela.geometry("350x200+100+100")
    janela.mainloop()

