#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:04:35 2018
@author: fabiosoaresv
"""

from Tkinter import *

class Dataset:
    janela = Tk()

    lb1 = Label(janela, text="Treino: ")
    lb2 = Label(janela, text="Teste: ")

    ed1 = Entry(janela,)
    ed2 = Entry(janela,)

    btn1 = Button(janela, text="Procurar")
    btn2 = Button(janela, text="Procurar")
    btn3 = Button(janela, text="Salvar")
    btn4 = Button(janela, text="Cancelar")

    lb1.grid(row=0, column=0)
    lb2.grid(row=2, column=0)
    ed1.grid(row=0, column=1)
    ed2.grid(row=2, column=1)
    btn1.grid(row=0, column=2)
    btn2.grid(row=2, column=2)
    btn3.grid(row=5, column=1)
    btn4.grid(row=7, column=1)
    
    janela.title("Importação do Dataset")
    janela.geometry("300x120+100+100")
    janela.mainloop()

