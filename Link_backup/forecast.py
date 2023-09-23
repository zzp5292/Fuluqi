# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:06:08 2023

@author: Lenovo2
"""


import os
import xlrd
import networkx as nx
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.layers import Dropout,Dense,GRU
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import math
from sklearn.model_selection import train_test_split
#import openpyxl    #读取=加载， 新建


def number(number,node):     #源地址与目的地址赋值    
    source_number = []
    for i in number:
        for j in node:
            if i == j:
               source_number.append(node.index(j)+1)
    return source_number    

def target_flow_addition(sources,targets,target_flow):   #计算目的节点流量总量
   
    flow = []
    target_allflow_sp = []
   # target_index = []
    target_sets = []
    tar = []
    target_sets = list(set(targets))       #源地址分类转化成列表形式
    targets_flow = list(zip(targets,target_flow))
    #print(len(set(sources)))               #部分节点无流量通过，因此事实节点少于50
    #print(len(set(targets)))
    i = 0
    j = 0
    allflow = 0
    for target1 in target_sets:
         allflow = 0
         for target2 in targets_flow:
             if target2[0] == target1:                 #匹配节点，相同节点流量相加            
                 allflow = allflow + target2[1]
                 flow.append(target_flow[j])               
         tar.append(target1)
         target_allflow_sp.append(allflow)
         
    target_allflow = list(zip(tar, target_allflow_sp))    
    return targets_flow,target_allflow
    
def allflow_sum(target_allflow):           #计算总流量
    
    flow = 0
    all = 0
    for flow in target_allflow:
        all = all + flow[1]
    return all

def standard(target_allflow):  #标准化公式
    min_flow = target_allflow[0][1]
    max_flow = target_allflow[0][1]
    tar = []
    target_allflow_standard = []
    
    for key in target_allflow:
        if key[1] < min_flow:
            min_flow = key[1]
        if key[1] > max_flow:
            max_flow = key[1]
    for key in target_allflow:
        p = (key[1] - min_flow)/(max_flow - min_flow)
        tar.append(key[0])
        target_allflow_standard.append(p)
    target_allflow_standard = list(zip(tar,target_allflow_standard))    
    
    return target_allflow_standard

if __name__ == "__main__":


    book1 = xlrd.open_workbook(r'~\germany01.xlsx')
    sheet_node = book1.sheets()[0]
    node_topo = sheet_node.col_values(6)[1:51] 
    
    #构建网络拓扑
    
    book_topo = xlrd.open_workbook(r'~\拓扑1.xlsx')
    sheet_topo = book_topo.sheets()[0]
    source_topo = sheet_topo.col_values(0)[0:]
    target_topo = sheet_topo.col_values(2)[0:]
    
    source_node = number(source_topo,node_topo)
    target_node = number(target_topo,node_topo)
    
    node_topo_number = number(node_topo,node_topo)    #将节点转化为数字
    link = list(zip(source_node,target_node))
       
    node_number = []
    j = 0
    for i in range(len(node_topo_number)):
        j = j+1
        node_number.append(j)
        
    edge_links = []
    j = 0
    for i in range(len(link)):
        j = j+1
        edge_links.append(link[i])
    
    G = nx.Graph()
    G.add_nodes_from(node_number)
    G.add_edges_from(edge_links)
    #nx.draw(G, with_labels = True,node_color = 'y',style = 'solid')
    
    
    

    data = []
    for i in range(176):
        data.append([])
    
    data_all = dict(zip(link,data))
    
    
    

    time = []
    F = []
    t = 0



    file_list = []
    path_1 = r"~\网络数据集"
    
    
    for file in os.listdir(path_1):
        t = t + 1
        time.append(t)
        f = []
        #wb = load_workbook(path+'\\'+file)
        #print(path_1+'\\'+file)
          
        wb = xlrd.open_workbook(path_1+'\\'+file)
        sheet1 = wb.sheets()[0]
        #sheet = wb['Sheet1']
     
   
        node_topo = sheet1.col_values(6)[1:51] 
   
        source1 = sheet1.col_values(11)[51:]
        target1 = sheet1.col_values(12)[51:]
        target_flow1 = sheet1.col_values(13)[51:]
   
        source1_number = number(source1,node_topo)
        target1_number = number(target1,node_topo)
    
        targets_flow,target_allflow1 = target_flow_addition(source1_number,target1_number,target_flow1)          
        allflow1 = allflow_sum(target_allflow1)



        target_allflow_standard = standard(target_allflow1)

   
        source_target1 = list(zip(source1_number,target1_number))
    #source_target2 = list(zip(source2,target2))
   
        path = []
        for path_link in source_target1:
            path.append(nx.dijkstra_path(G,path_link[0],path_link[1],1))      #计算转发路径
   

        path_link_flow = list(zip(path,target_flow1))
    
    
    

        flow = []
        for i in range(len(link)):
            flow.append([])
        link_flow_split = dict(zip(link,flow))
    
        flow_all = []
        for i in range(len(link)):
            flow_all.append(0)
        link_flow_all = dict(zip(link,flow_all)) 
    

       
        for path_single,flow_single in path_link_flow:
            for i in range(len(path_single) - 1):
                link_flow_split[path_single[i],path_single[i + 1]].append(flow_single)
                link_flow_all[path_single[i],path_single[i + 1]] = link_flow_all[path_single[i],path_single[i + 1]] + flow_single
        
       
        
        for key,value in link_flow_all.items():
            data_all[key].append(value)
            f.append(value)
        F.append(f)
    All = dict(zip(time, F))




    train_all = []
    train_set = []
    train_test = []
    train_lable = []
    

    for key,value in All.items():
        for flow_value in value:
            train_lable.append(key)
            train_all.append(flow_value)


    train_set  = train_all[0:4048]
    train_test = train_all[4048:4928]

    train_set = pd.DataFrame(train_set)
    train_test = pd.DataFrame(train_test)

    k_test = train_test.copy()



    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_set)
    train_test = sc.transform(train_test) 

    
    x_train = []
    y_train = []
 
    x_test = []
    y_test = []
    
    for i in range(176, len(training_set_scaled)):
        x_train.append(training_set_scaled[i - 176:i, 0])
        y_train.append(training_set_scaled[i, 0])


    np.random.seed(7)
    np.random.shuffle(x_train)
    np.random.seed(7)
    np.random.shuffle(y_train)
    tf.compat.v1.random.set_random_seed(7)


    x_train, y_train = np.array(x_train), np.array(y_train)


    x_train = np.reshape(x_train, (x_train.shape[0], 176, 1))
    
    

    for i in range(176, len(train_test)):
        x_test.append(train_test[i - 176:i, 0])
        y_test.append(train_test[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    x_test = np.reshape(x_test, [x_test.shape[0], 176, 1])
    

    model = tf.keras.Sequential([
            GRU(80, return_sequences=True),  # return_sequences=True，循环核各时刻会把ht推送到下一层
            Dropout(0.2),
            GRU(100),
            Dropout(0.2),
            Dense(1)
        ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mean_squared_error'  # 损失函数用均方误差
    )

    checkpoint_path_save = './checkpoint/gru_stock.ckpt'

    if os.path.exists(checkpoint_path_save + '.index'):
       print('--------------------load the model----------------------')

       model.load_weights(checkpoint_path_save)




    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_save,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )



    history = model.fit(x_train, y_train,
                    batch_size=176, epochs=25, validation_data=(x_test, y_test),
                    validation_freq=5, callbacks=[cp_callback])
 
# 统计网络结构参数
    model.summary()



    file1 = open('./gru_weights_stock.txt', 'w')
    for v in model.trainable_variables:



        file1.write(str(v.name) + '\n')
        file1.write(str(v.shape) + '\n')
        file1.write(str(v.numpy()) + '\n')
    file1.close()
 
# 获取loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
 
# 绘制loss
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show() 
    
    
    

    predicted_stock_price = model.predict(x_test)

    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    real_stock_price = sc.inverse_transform(train_test[176:])
    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(real_stock_price, color='red', label='Real link flow')
    plt.plot(predicted_stock_price, color='blue', label='Predicted link flow')
    plt.xlabel('Time', fontdict={'family': 'Times New Roman', 'size': 30})  # 设置标签的字体
    plt.ylabel('link flow', fontdict={'family': 'Times New Roman', 'size': 35})
    plt.yticks(fontproperties='Times New Roman', size=20)  # 设置刻度的字体及字号大小
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.show()
    


    mse = mean_squared_error(real_stock_price, predicted_stock_price)
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    mae = mean_absolute_error(real_stock_price, predicted_stock_price)
    r2 = r2_score(real_stock_price, predicted_stock_price)

    print('均方误差: %.6f' % mse)
    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)
    print('R2: %.6f' % r2)

