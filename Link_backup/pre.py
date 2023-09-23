import os
import xlrd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

book1 = xlrd.open_workbook(r'~\up.xlsx')
sheet_node = book1.sheets()[0]
GRU = sheet_node.col_values(0)[0:]
Test = sheet_node.col_values(1)[0:]
SVR = sheet_node.col_values(2)[0:]

fig = plt.figure(figsize=(16, 9))
size = 4
x = np.arange(size)

total_width, n = 0.5, 3
width = total_width / n

x = x - (total_width - width) / 2 + 25
print(x)

plt.bar(x, SVR, width=width, color='orange',  label="SVR")
plt.bar(x + width, Test, width=width, color='blue', label="Test")
plt.bar(x + 2*width, GRU, width=width, color='red', label="GRU")
plt.xlabel('time', fontdict={'family': 'Times New Roman', 'size': 25})  # 设置标签的字体
plt.ylabel('number of links', fontdict={'family': 'Times New Roman', 'size': 25})
plt.rcParams.update({'font.size': 15})
plt.legend(loc='upper right')


plt.yticks(fontproperties='Times New Roman', size=20)  # 设置刻度的字体及字号大小
plt.xticks(fontproperties='Times New Roman', size=20)
plt.legend()
plt.show()
