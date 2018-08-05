#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:18:30 2018
@author: fabiosoaresv
"""
from Tkinter import *

class Login:
    janela = Tk()
    
    lb1 = Label(janela, text="Informações do Dataset: ")
    lb2 = Label(janela, text="Quantidade de dados: ")
    lb3 = Label(janela, text="Quantidade para treino: ")
    lb4 = Label(janela, text="Tamanho do Dataset: ")
    lb5 = Label(janela, text="Quantidade para teste: ")
    
    ed1 = Entry(janela,)
    ed2 = Entry(janela,)
    ed3 = Entry(janela,)
    ed4 = Entry(janela,)
    
    btn1 = Button(janela, text="-")
    btn2 = Button(janela, text="+")

    lb1.grid(row=0, column=0)
    lb2.grid(row=1, column=0)
    lb3.grid(row=1, column=2)
    lb4.grid(row=1, column=0)
    lb5.grid(row=1, column=0)
    ed1.grid(row=1, column=4)
    ed2.grid(row=1, column=1)
    ed3.grid(row=1, column=1)
    ed4.grid(row=1, column=1)
    btn1.grid(row=1, column=3)
    btn2.grid(row=1, column=5)

    janela.title("Tipo de Rede")
    janela.geometry("800x500+100+100")
    janela.mainloop()

