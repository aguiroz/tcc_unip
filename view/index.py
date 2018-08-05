#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:17:35 2018
@author: fabiosoaresv
"""
from Tkinter import *
#import tkinter.messagebox as tkMessagebox

class Index:
    janela = Tk()

    lb0 = Label(janela, text="Seja Bem-Vindo! :) \n Esse é o nosso Software, \nvamos formular uma explicação melhor haha")
    lb1 = Label(janela, text="Dataset: ")
    lb2 = Label(janela, text="Feedforward: ")
    lb3 = Label(janela, text="RNN: ")
    lb4 = Label(janela, text="CNN: ")
    lb5 = Label(janela, text="Parâmetros: ")
    lb6 = Label(janela, text="Estatísticas: ")
    lb7 = Label(janela, text="Explicação")
   
    # Esse botões devem carregar o dataset.py
    btn1 = Button(janela, text="Carregar")
    btn2 = Button(janela, text="Carregar")
    btn3 = Button(janela, text="Carregar")
    btn4 = Button(janela, text="Carregar")
    btn5 = Button(janela, text="Carregar")
    btn6 = Button(janela, text="Carregar")
    btn7 = Button(janela, text="Sair", command=janela.destroy)

    lb0.grid(row=5, column=0)
    lb1.grid(row=0, column=0)
    lb2.grid(row=0, column=4)
    lb3.grid(row=0, column=6)
    lb4.grid(row=1, column=0)
    lb5.grid(row=1, column=4)
    lb6.grid(row=1, column=6)
    
    btn1.grid(row=0, column=2)
    btn2.grid(row=0, column=5)
    btn3.grid(row=0, column=8)
    btn4.grid(row=1, column=2)
    btn5.grid(row=1, column=5)
    btn6.grid(row=1, column=8)
    btn7.grid(row=16, column=16)

    janela.title("Home")
    janela.geometry("750x200+100+100")
    janela.mainloop()

