
import os
import xlrd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

book1 = xlrd.open_workbook(r'~\resultes.xlsx')
sheet_node = book1.sheets()[0]
predicted_stock_price = sheet_node.col_values(0)[0:]
for key in predicted_stock_price:
    i = i + 1
    if 1 <= i < 177:
        Y_P1.append(key)
    elif 177 <= i < 353:
        Y_P2.append(key)
    elif 353 <= i < 529:
        Y_P3.append(key)
    elif 529 <= i < 705:
        Y_P4.append(key)
max1 = max(Y_P1)
max2 = max(Y_P2)
max3 = max(Y_P3)
max4 = max(Y_P4)
Y1 = dict(zip(link, Y_P1))
Y2 = dict(zip(link, Y_P2))
Y3 = dict(zip(link, Y_P3))
Y4 = dict(zip(link, Y_P4))
over1 = []
over2 = []
over3 = []
over4 = []
t = 0.7
for key, value in Y1.items():
    c = value / max1
    if c > t:
        over1.append(key)
for key, value in Y2.items():
    c = value / max2
    if c > t:
        over2.append(key)
for key, value in Y3.items():
    c = value / max3
    if c > t:
        over3.append(key)
for key, value in Y4.items():
    c = value / max4
    if c > t:
        over4.append(key)
print("update note")
print(over1)
print(over2)
print(over3)
print(over4)