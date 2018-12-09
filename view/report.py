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
    lb1 = Label(janela, text="Rede Neural Multicamadas ")
    lb2 = Label(janela, text="Custo Final: ")
    lb3 = Label(janela, text="Taxa de Acerto: ")
    lb4 = Label(janela, text="Qtde de Acerto: ")
    lb5 = Label(janela, text="Tempo Decorrido: ")
    lb6 = Label(janela, text="Rede Neural Recorrente ")
    lb7 = Label(janela, text="Custo Final: ")
    lb8 = Label(janela, text="Taxa de Acerto: ")
    lb9 = Label(janela, text="Qtde de Acerto: ")
    lb10 = Label(janela, text="Tempo Decorrido: ")
    lb11 = Label(janela, text="Rede Neural Convolucional ")
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
    lb23 = Label(janela, text=" ")
    lb24 = Label(janela, text=" ")
    lb25 = Label(janela, text=" ")
    lb26 = Label(janela, text="Legenda:")
    lb27 = Label(janela, text="Vermelho - Rede Neural Multicamadas ")
    lb28 = Label(janela, text="Verde - Rede Neural Convolucional")
    lb29 = Label(janela, text="Azul - Rede Neural Recorrente")

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

    # Label's de Legenda
    lb26.grid(row=25, column=2)
    lb27.grid(row=26, column=2)
    lb28.grid(row=27, column=2)
    lb29.grid(row=28, column=2)

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
    lb23.grid(row=22, column=2)
    lb24.grid(row=23, column=2)
    lb25.grid(row=24, column=2)

    janela.title("Comparativo das Arquiteturas das Rede Neurais")
    janela.geometry("1220x700+100+100")
    janela.mainloop()
