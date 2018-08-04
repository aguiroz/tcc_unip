#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Ago 4 18:17:10 2018
@author: fabiosoaresv
"""
from Tkinter import *

class Login:
    janela = Tk()

    lb1 = Label(janela, text="Login: ")
    lb2 = Label(janela, text="Senha: ")
    
    ed1 = Entry(janela,)
    ed2 = Entry(janela,)

    btn1 = Button(janela, text="Entrar")

    lb1.grid(row=0, column=0)
    lb2.grid(row=1, column=0)
    ed1.grid(row=0, column=1)
    ed2.grid(row=1, column=1)
    btn1.grid(row=2, column=1)

    janela.title("Login")
    janela.geometry("220x70+100+100")
    janela.mainloop()

