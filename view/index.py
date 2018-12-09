#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:17:35 2018
@author: fabiosoaresv
"""
from Tkinter import *

class Index:
    janela = Tk()

    lb0 = Label(janela, text="Seja Bem-Vindo ao MatPy! :)")
    lb1 = Label(janela, text="Importar Dataset ")
    lb2 = Label(janela, text="RNA - Multicamadas ")
    lb3 = Label(janela, text="RNA - Recorrente")
    lb4 = Label(janela, text="RNA - Convolucional")
    lb5 = Label(janela, text="Selecionar Algoritmo de Treino")
    lb6 = Label(janela, text="Visualizar Comparativos")
    lb7 = Label(janela, text="Explicação")

    # Esse botões devem carregar o dataset.py
    btn1 = Button(janela, text="Carregar")
    btn2 = Button(janela, text="Carregar")
    btn3 = Button(janela, text="Carregar")
    btn4 = Button(janela, text="Carregar")
    btn5 = Button(janela, text="Carregar")
    btn6 = Button(janela, text="Carregar")
    btn7 = Button(janela, text="Sair", command=janela.destroy)

    lb0.grid(row=0, column=0)
    lb1.grid(row=1, column=0)
    lb2.grid(row=1, column=1)
    lb3.grid(row=3, column=1)
    lb4.grid(row=6, column=1)
    lb5.grid(row=3, column=0)
    lb6.grid(row=1, column=2)

    btn1.grid(row=2, column=0)
    btn2.grid(row=2, column=1)
    btn3.grid(row=4, column=1)
    btn4.grid(row=7, column=1)
    btn5.grid(row=4, column=0)
    btn6.grid(row=2, column=2)
    btn7.grid(row=16, column=16)

    janela.title("MatPy - Ciência da Computação - UNIP 2018")
    janela.geometry("600x200+100+100")
    janela.mainloop()
