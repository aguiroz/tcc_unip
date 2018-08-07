#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:18:30 2018
@author: fabiosoaresv
"""
from Tkinter import *
import matplotlib.pyplot as plt
import numpy as np

class Report:
    janela = Tk()
    
    lb0 = Label(janela, text="Estatísticas ")
    lb1 = Label(janela, text="Feedforward ")
    lb2 = Label(janela, text="Custo Final: ")
    lb3 = Label(janela, text="Taxa de Acerto: ")
    lb4 = Label(janela, text="Qtde de Acerto: ")
    lb5 = Label(janela, text="Tempo Decorrido: ")
    lb6 = Label(janela, text="RNN ")
    lb7 = Label(janela, text="Custo Final: ")
    lb8 = Label(janela, text="Taxa de Acerto: ")
    lb9 = Label(janela, text="Qtde de Acerto: ")
    lb10 = Label(janela, text="Tempo Decorrido: ") 
    lb11 = Label(janela, text="CNN ")
    lb12 = Label(janela, text="Custo Final: ")
    lb13 = Label(janela, text="Taxa de Acerto: ")
    lb14 = Label(janela, text="Qtde de Acerto: ")
    lb15 = Label(janela, text="Tempo Decorrido: ")
    lb16 = Label(janela, text=" ")
    lb17 = Label(janela, text=" ")
    lb18 = Label(janela, text=" ")
    lb19 = Label(janela, text="Gráficos ")
    lb20 = Label(janela, text="Gráfico Tempo x Iteração: ")
    lb21 = Label(janela, text="Gráfico Custo x Acerto: ")
    lb22 = Label(janela, text="Gráfico Custo x Iteração: ")
    
    ed1 = Entry(janela,)
    ed2 = Entry(janela,)
    ed3 = Entry(janela,)
    ed4 = Entry(janela,)
    ed5 = Entry(janela,)
    ed6 = Entry(janela,)
    ed7 = Entry(janela,)
    ed8 = Entry(janela,)
    ed9 = Entry(janela,)
    ed10 = Entry(janela,)
    ed11 = Entry(janela,)
    ed12 = Entry(janela,)

    btn1 = Button(janela, text="Gerar")
    btn2 = Button(janela, text="Gerar")
    btn3 = Button(janela, text="Gerar")
    
    # Label's de divisão
    lb16.grid(row=3, column=0)

    # Feedforward    
    lb1.grid(row=4, column=0)
    lb2.grid(row=6, column=0)
    lb3.grid(row=7, column=0)
    lb4.grid(row=8, column=0)
    lb5.grid(row=9, column=0)

    # RNN
    lb6.grid(row=4, column=2)
    lb7.grid(row=6, column=2)
    lb8.grid(row=7, column=2)
    lb9.grid(row=8, column=2)
    lb10.grid(row=9, column=2)
 
    # CNN
    lb11.grid(row=4, column=4)
    lb12.grid(row=6, column=4)
    lb13.grid(row=7, column=4)
    lb14.grid(row=8, column=4)
    lb15.grid(row=9, column=4)
    
    # Label's de divisão
    lb17.grid(row=12, column=0)
    lb18.grid(row=15, column=0)
    
    ed1.grid(row=6, column=1)
    ed2.grid(row=7, column=1)
    ed3.grid(row=8, column=1)
    ed4.grid(row=9, column=1)
    ed5.grid(row=6, column=3)
    ed6.grid(row=7, column=3)
    ed7.grid(row=8, column=3)
    ed8.grid(row=9, column=3)
    ed9.grid(row=6, column=5)
    ed10.grid(row=7, column=5)
    ed11.grid(row=8, column=5)
    ed12.grid(row=9, column=5)
    
    # Label's de divisão
    lb19.grid(row=16, column=2)
    lb20.grid(row=17, column=2)
    btn1.grid(row=17, column=3)
    lb21.grid(row=18, column=2)
    btn2.grid(row=18, column=3)
    lb22.grid(row=19, column=2)
    btn3.grid(row=19, column=3)

    janela.title("Tipo de Rede")
    janela.geometry("900x300+100+100")
    janela.mainloop()

