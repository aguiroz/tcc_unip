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
    lb6 = Label(janela, text="Treino ")
    lb7 = Label(janela, text="Custo: ")
    lb8 = Label(janela, text="Erro: ")
    lb9 = Label(janela, text="Acerto: ")
    lb10 = Label(janela, text="Interação: ")
    lb11 = Label(janela, text="Lote: ")
    lb12 = Label(janela, text="Teste ")
    lb13 = Label(janela, text="Custo: ")
    lb14 = Label(janela, text="Erro: ")
    lb15 = Label(janela, text="Acerto: ")
    lb16 = Label(janela, text="Tempo: ")
    lb17 = Label(janela, text="Barra de progresso: ")
    lb18 = Label(janela, text="Gráfico Treino x Teste: ")
    lb19 = Label(janela, text=" ")
    lb20 = Label(janela, text=" ")
    lb21 = Label(janela, text=" ")
    lb22 = Label(janela, text=" ")
    
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
    ed13 = Entry(janela,)
    ed14 = Entry(janela,)
    
    btn1 = Button(janela, text="-")
    btn2 = Button(janela, text="+")
    btn3 = Button(janela, text="-")
    btn4 = Button(janela, text="+")
    btn5 = Button(janela, text="Treinar")
    btn6 = Button(janela, text="Testar / Prever")
    
    lb1.grid(row=0, column=0)
    lb2.grid(row=1, column=0)
    lb3.grid(row=1, column=2)
    lb4.grid(row=2, column=0)
    lb5.grid(row=2, column=2)
    
    # Label's de divisão
    lb19.grid(row=3, column=0)
    lb20.grid(row=4, column=0)
    
    lb6.grid(row=5, column=0)
    lb7.grid(row=6, column=0)
    lb8.grid(row=7, column=0)
    lb9.grid(row=8, column=0)
    lb10.grid(row=9, column=0)
    lb11.grid(row=10, column=0)
    lb12.grid(row=5, column=2)
    lb13.grid(row=6, column=2)
    lb14.grid(row=7, column=2)
    lb15.grid(row=8, column=2)
    lb16.grid(row=9, column=2)
    lb17.grid(row=16, column=0)
    
    # Label's de divisão
    lb21.grid(row=12, column=0)
    lb22.grid(row=15, column=0)
    
    ed1.grid(row=1, column=1)
    ed2.grid(row=2, column=1)
    ed3.grid(row=1, column=4)
    ed4.grid(row=2, column=4)
    ed5.grid(row=6, column=1)
    ed6.grid(row=7, column=1)
    ed7.grid(row=8, column=1)
    ed8.grid(row=9, column=1)
    ed9.grid(row=10, column=1)
    ed10.grid(row=6, column=3)
    ed11.grid(row=7, column=3)
    ed12.grid(row=8, column=3)
    ed13.grid(row=9, column=3)
    ed14.grid(row=16, column=1)
    
    btn1.grid(row=1, column=3)
    btn2.grid(row=1, column=5)
    btn3.grid(row=2, column=3)
    btn4.grid(row=2, column=5)
    btn5.grid(row=14, column=1)
    btn6.grid(row=14, column=3)

    janela.title("Tipo de Rede")
    janela.geometry("900x450+100+100")
    janela.mainloop()

