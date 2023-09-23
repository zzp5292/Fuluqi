import os
import xlrd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

book1 = xlrd.open_workbook(r'~\update.xlsx')
sheet_node = book1.sheets()[0]
GRU = sheet_node.col_values(0)[0:]
Test = sheet_node.col_values(1)[0:]
SVR = sheet_node.col_values(2)[0:]
fig = plt.figure(figsize=(16, 9))
size = 4

total_width, n = 0.5, 3
width = total_width / n

x = np.arange(size)
x = x  + 25

plt.plot(x, GRU, 'r', marker='o', alpha=1, linewidth=2, label="GRU")  # '
plt.plot(x, SVR, 'g', marker='o', alpha=1, linewidth=2, label="SVR")



plt.xlabel('time', fontdict={'family': 'Times New Roman', 'size': 25})  # 设置标签的字体
plt.ylabel('accuracy', fontdict={'family': 'Times New Roman', 'size': 25})
plt.rcParams.update({'font.size': 15})
plt.legend(loc='upper right')

plt.yticks(fontproperties='Times New Roman', size=20)  # 设置刻度的字体及字号大小
plt.xticks(fontproperties='Times New Roman', size=20)
plt.legend()
plt.show()
