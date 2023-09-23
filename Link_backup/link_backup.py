import pandas as pd
import xlrd
# import xlwt
import numpy as np
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy


def remove1(source):

    for i in source:
        source.remove("")
        source.remove('')


def number(number, node):
    source_number = []
    for i in number:
        for j in node:
            if i == j:
                source_number.append(node.index(j) + 1)
    return source_number


def topNBetweeness(G):
    score = nx.betweenness_centrality(G)
    score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    output = []
    for node in score:
        output.append(node[0])
    print(output)
    fout = open("betweennessSorted.data", 'w')
    for target in output:
        fout.write(str(target) + " ")
    return score


def addEdge(a, b):
    global edgeLinks
    if a not in edgeLinks:
        edgeLinks[a] = set()
    if b not in edgeLinks:
        edgeLinks[b] = set()
    edgeLinks[a].add(b)
    edgeLinks[b].add(a)


def target_flow_addition(sources, targets, target_flow):

    flow = []
    target_allflow_sp = []
    target_sets = []
    tar = []
    target_sets = list(set(targets))
    targets_flow = list(zip(targets, target_flow))
    print(len(set(sources)))
    print(len(set(targets)))
    i = 0
    j = 0
    allflow = 0
    for target1 in target_sets:
        allflow = 0
        for target2 in targets_flow:
            if target2[0] == target1:
                allflow = allflow + target2[1]
                flow.append(target_flow[j])
        tar.append(target1)
        target_allflow_sp.append(allflow)

    target_allflow = list(zip(tar, target_allflow_sp))
    return targets_flow, target_allflow


def allflow_sum(target_allflow):

    flow = 0
    all = 0
    for flow in target_allflow:
        all = all + flow[1]
    return all


def Hi_pi(target_flow, target_allflow):

    Hi1 = []
    Hi2 = []
    Pi = []
    tar = []
    tar1 = []
    hi = 0
    node_sort = []
    max_flow = 0
    min_flow = 0

    for node in target_allflow:
        node_sort.append(node[0])

    for i in target_allflow:
        for j in target_flow:
            if i[0] == j[0]:
                pi = j[1] / i[1]
                Pi.append(pi)
                tar.append(j[0])

    target_pi = list(zip(tar, Pi))

    for node in node_sort:
        hi = 0
        for key in target_pi:
            if node == key[0]:
                p = (-1) * (key[1]) * math.log(key[1])
                hi = hi + p
        tar1.append(node)
        Hi1.append(hi)
    for key in Hi1:
        if max_flow < key:
            max_flow = key
        if min_flow > key:
            min_flow = key
    for key in Hi1:
        p = (key - min_flow) / (max_flow - min_flow)

        Hi2.append(p)
    target_hi = list(zip(tar1, Hi2))
    return target_pi, target_hi


def GNT(h1, h2, p1, p2):
    h = []
    p = []
    GNT = []
    tar = []
    tar2 = []

    for k1 in h1:
        for k2 in h2:
            if k1[0] == k2[0]:
                h.append(k2[1] - k1[1])
        tar.append(k1[0])

    h = list(zip(tar, h))

    for k1 in p1:
        for k2 in p2:
            if k1[0] == k2[0]:
                p.append(k2[1] - k1[1])
        tar2.append(k1[0])
    p = list(zip(tar2, p))

    for k1 in h:
        for k2 in p:
            if k1[0] == k2[0]:
                if k1[1] == 0:
                    GNT.append(k2[1])
                else:
                    GNT.append(k2[1] / k1[1])
    GNT = list(zip(tar2, GNT))
    return p, h, GNT


def topo1_entropy(betweenness_centrality):
    p = 0
    target = []
    To1 = []
    To2 = []
    all_topo = 0
    min_topo = betweenness_centrality[0][1]
    max_topo = betweenness_centrality[0][1]

    for key in betweenness_centrality:
        all_topo = all_topo + key[1]

    for key in betweenness_centrality:
        target.append(key[0])
        p = ((key[1] / all_topo) + 1) * math.log((key[1] / all_topo) + 1)
        if min_topo > p:
            min_topo = p
        if max_topo < p:
            max_topo = p
        To1.append(p)
    for key in To1:
        p = (key - min_topo) / (max_topo - min_topo)
        To2.append(p)
    topo_dict = list(zip(target, To2))
    return topo_dict


def degree_sum(flow_entropy, GNT, topo_entropy):
    a = 0.0
    b = 0.3
    c = 0.7
    k = 0
    tar = []
    degree = []
    note_degree = []
    flow = flow_entropy.copy()
    GNT1 = GNT.copy()
    topo1 = topo_entropy.copy()
    len1 = len(topo1)
    for k in range(len1):
        if flow[k][0] != topo1[k][0]:
            flow.insert(k, (topo1[k][0], 0))
    for k in range(len1):
        if GNT1[k][0] != topo1[k][0]:
            GNT1.insert(k, (topo1[k][0], 0))
    for k in range(len1):
        if flow[k][0] == topo1[k][0] and topo1[k][0] == GNT1[k][0]:
            p = a * flow[k][1] + b * topo1[k][1] + c * GNT1[k][1]
        tar.append(topo1[k][0])
        degree.append(p)

    node_degree = dict(zip(tar, degree))

    q = 0.3
    p = 0.5
    o = 0.2
    len2 = len(node_degree)
    j = 1
    node_degree = sorted(node_degree.items(), key=lambda item: item[1])
    tar1 = []
    degree1 = []
    for node in node_degree:
        if j <= o * len2:
            tar1.append(node[0])
            degree1.append(1)
        elif j > o * len2 and j <= (o + p) * len2:
            tar1.append(node[0])
            degree1.append(2)
        else:
            tar1.append(node[0])
            degree1.append(3)
        j = j + 1

    node_degree1 = list(zip(tar1, degree1))

    return node_degree1


def link_sum(edge_links, node_degree):
    k = 0
    i = 0
    ol = []
    ol1 = []
    link_degree = []
    link_sort1 = edge_links
    for k in range(int(len(edge_links))):
        i = i + 1
        for j in range(i, int(len(edge_links))):
            if link_sort1[k][0] == link_sort1[j][1] and link_sort1[k][1] == link_sort1[j][0]:
                l = [link_sort1[k], link_sort1[j]]
                ol.append(l)
                ol1.append(link_sort1[k])

    k = 0
    weight = 0
    for node in ol1:
        i = node[0]
        j = node[1]
        k1 = node_degree[i - 1][1]
        k2 = node_degree[j - 1][1]
        k = k1 + k2
        if k < 4:
            weight = 1
            link_degree.append(weight)
        elif k >= 4 and k < 6:
            weight = 2
            link_degree.append(weight)
        elif k == 6:
            weight = 3
            link_degree.append(weight)

    link_degree = list(zip(ol1, link_degree))

    return link_degree


def standard(target_allflow):
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
        p = (key[1] - min_flow) / (max_flow - min_flow)
        tar.append(key[0])
        target_allflow_standard.append(p)
    target_allflow_standard = list(zip(tar, target_allflow_standard))

    return target_allflow_standard


def link_degree_order(link_degree):
    order = link_degree.copy()
    list_len = len(link_degree)
    a = 0.3
    b = 0.5
    i = 0
    tar = []
    degree = []
    order.sort(key=lambda ele: ele[1], reverse=False)
    for key in order:
        if i < int(a * list_len):
            tar.append(key[0])
            degree.append(1)
        elif i < int((a + b) * list_len):
            tar.append(key[0])
            degree.append(2)
        else:
            tar.append(key[0])
            degree.append(3)
        i = i + 1
        order = list(zip(tar, degree))
    return order


def Dijkstra(network, s, d):
    path = []
    n = len(network)
    fmax = 9999999
    w = [[0 for i in range(n)] for j in range(n)]
    book = [0 for i in range(n)]
    dis = [fmax for i in range(n)]
    book[s - 1] = 1
    midpath = [-1 for i in range(n)]
    u = s - 1
    for i in range(n):
        for j in range(n):
            if network[i][j] != 0:
                w[i][j] = network[i][j]
            else:
                w[i][j] = fmax
            if i == s - 1 and network[i][j] != 0:
                dis[j] = network[i][j]
    for i in range(n - 1):
        min = fmax
        for j in range(n):
            if book[j] == 0 and dis[j] < min:
                min = dis[j]
                u = j
        book[u] = 1
        for v in range(n):
            if dis[v] > dis[u] + w[u][v]:
                dis[v] = dis[u] + w[u][v]
                midpath[v] = u + 1
    j = d - 1
    path.append(d)
    while (midpath[j] != -1):
        path.append(midpath[j])
        j = midpath[j] - 1
    path.append(s)
    path.reverse()
    return path


def return_path_sum(network, path):
    result = 0
    for i in range(len(path) - 1):
        result += network[path[i] - 1][path[i + 1] - 1]
    return result


def add_limit(path, s):
    result = []
    for item in path:
        if s in item[0]:
            result.append([s, item[0][item[0].index(s) + 1]])
    result = [list(r) for r in list(set([tuple(t) for t in result]))]
    return result


def return_shortest_path_with_limit(network, s, d, limit_segment, choice):
    mid_net = copy.deepcopy(network)
    for item in limit_segment:
        mid_net[item[0] - 1][item[1] - 1] = mid_net[item[1] - 1][item[0] - 1] = 0
    s_index = choice.index(s)
    for point in choice[:s_index]:
        for i in range(len(mid_net)):
            mid_net[point - 1][i] = mid_net[i][point - 1] = 0
    mid_path = Dijkstra(mid_net, s, d)
    return mid_path


def judge_path_legal(network, path):
    for i in range(len(path) - 1):
        if network[path[i] - 1][path[i + 1] - 1] == 0:
            return False
    return True


def k_shortest_path(network, s, d, k):
    k_path = []
    alter_path = []
    kk = Dijkstra(network, s, d)
    k_path.append([kk, return_path_sum(network, kk)])
    while (True):
        if len(k_path) == k: break
        choice = k_path[-1][0]
        for i in range(len(choice) - 1):
            limit_path = [[choice[i], choice[i + 1]]]
            if len(k_path) != 1:
                limit_path.extend(add_limit(k_path[:-1], choice[i]))
            mid_path = choice[:i]
            mid_res = return_shortest_path_with_limit(network, choice[i], d, limit_path, choice)
            if judge_path_legal(network, mid_res):
                mid_path.extend(mid_res)
            else:
                continue
            mid_item = [mid_path, return_path_sum(network, mid_path)]
            if mid_item not in k_path and mid_item not in alter_path:
                alter_path.append(mid_item)
        if len(alter_path) == 0:
            print("总共只有{}条最短路径！".format(len(k_path)))
            return k_path
        alter_path.sort(key=lambda x: x[-1])
        x = alter_path[0][-1]
        y = len(alter_path[0][0])
        u = 0
        for i in range(len(alter_path)):
            if alter_path[i][-1] != x:
                break
            if len(alter_path[i][0]) < y:
                y = len(alter_path[i][0])
                u = i
        k_path.append(alter_path[u])
        alter_path.pop(u)
    return k_path


if __name__ == "__main__":


    book = xlrd.open_workbook(r'~\germany01.xlsx')
    book4 = xlrd.open_workbook(r'~\拓扑1.xlsx')

    sheet1 = book.sheets()[0]
    sheet2 = book4.sheets()[0]

    node_topo = []
    source_topo = []
    target_topo = []

    node_topo = sheet1.col_values(6)[1:51]
    source_topo = sheet2.col_values(0)[0:]
    target_topo = sheet2.col_values(2)[0:]

    source_number = number(source_topo, node_topo)
    target_number = number(target_topo, node_topo)

    link = list(zip(source_number, target_number))
    link_sort = list(set(link))
    link_sort_len = len(set(link_sort))

    node_number = []
    j = 0
    for i in range(len(node_topo)):
        j = j + 1
        node_number.append(j)

    edge_links = []
    j = 0
    for i in range(len(link)):
        j = j + 1
        edge_links.append(link[i])

    llll = len(link)


    G = nx.Graph()
    G.add_nodes_from(node_number)
    G.add_edges_from(edge_links)
    G_array = np.array(nx.adjacency_matrix(G).todense())



    betweenness_centrality = []
    betweenness_centrality = topNBetweeness(G)

    To = topo1_entropy(betweenness_centrality)


    book5 = xlrd.open_workbook(r'~\germany24.xlsx')
    book6 = xlrd.open_workbook(r'~\germany25.xlsx')

    sheet3 = book5.sheets()[0]
    sheet4 = book6.sheets()[0]

    nrows = sheet3.nrows

    ncols = sheet3.ncols

    target_allflow = []
    allflow = 0
    target1 = 0
    allflow2 = 0


    source = sheet3.col_values(11)[51:]
    target = sheet3.col_values(12)[51:]
    target_flow = sheet3.col_values(13)[51:]

    source2 = sheet4.col_values(11)[51:]
    target2 = sheet4.col_values(12)[51:]
    target_flow2 = sheet4.col_values(13)[51:]

    print("target", len(set(target)))
    print("target2", len(set(target2)))

    source1_number = number(source, node_topo)
    target1_number = number(target, node_topo)
    source2_number = number(source2, node_topo)
    target2_number = number(target2, node_topo)

    targets_flow, target_allflow = target_flow_addition(source1_number, target1_number, target_flow)
    allflow = allflow_sum(target_allflow)

    targets_flow2, target_allflow2 = target_flow_addition(source2_number, target2_number, target_flow2)
    allflow2 = allflow_sum(target_allflow2)

    target_allflow_standard = standard(target_allflow)
    target_allflow2_standard = standard(target_allflow2)


    Hi, H = Hi_pi(targets_flow, target_allflow)

    Hi2, H2 = Hi_pi(targets_flow2, target_allflow2)  # Hi为信息熵

    h, p, GNT = GNT(H, H2, target_allflow_standard, target_allflow2_standard)

    GNT = standard(GNT)

    flow_entropy = H2.copy()
    topo_entropy = To.copy()
    flow_entropy = standard(flow_entropy)
    flow_entropy.sort(key=lambda ele: ele[0], reverse=False)
    topo_entropy.sort(key=lambda ele: ele[0], reverse=False)
    GNT.sort(key=lambda ele: ele[0], reverse=False)

    node_degree = degree_sum(flow_entropy, GNT, topo_entropy)

    link_degree = []
    link_degree = link_sum(link, node_degree)
    link_degree = link_degree_order(link_degree)

    G1 = nx.Graph()
    G1.add_nodes_from(node_number)
    for key in link_degree:
        G1.add_edge(key[0][0], key[0][1], weight=key[1])

    fig = plt.figure(figsize=(16, 9))
    pos = nx.spring_layout(G)


    weights = nx.get_edge_attributes(G1, "weight")

    nx.draw_networkx(G1, pos, with_labels=True, node_color='aquamarine')

    nx.draw_networkx_edge_labels(G1, pos, edge_labels=weights)

    plt.show()



    d_path_list = []
    d_path_length = []

    for i in range(1, 51):
        for j in range(i + 1, 51):
            s_path = nx.shortest_path(G, i, j, None)
            length = nx.shortest_path_length(G, i, j, None)
            d_path_list.append(s_path)
            d_path_length.append(length)




    count_source_target = list(zip(source2_number, target2_number))
    for key in count_source_target:
        s_path3 = nx.shortest_path(G5, key[0], key[1], None)
        for key1 in range(0, len(s_path3) - 1):
            link_IML = (s_path3[key1], s_path3[key1 + 1])
            for key2, key3 in link_degree_IML.items():
                if key2[0] == link_IML[0] and key2[1] == link_IML[1]:
                    link_degree_IML[key2] = link_degree_IML[key2] + 1  # 流数量

    l_j = list(zip(source2_number, target2_number))
    l_j_1 = dict(zip(l_j, target_flow2))

    for key, key1 in l_j_1.items():
        s_path4 = nx.shortest_path(G5, key[0], key[1], None)
        for key2 in range(0, len(s_path4) - 1):
            link_IML = (s_path4[key2], s_path4[key2 + 1])
            for key3, key4 in link_degree_flow_common.items():
                if key3[0] == link_IML[0] and key3[1] == link_IML[1]:
                    link_degree_flow_common[key3] = link_degree_flow_common[key3] + l_j_1[key]  # 流的大小

    for key1, value1 in link_degree_IML.items():
        for key2, value2 in link_degree_IML.items():
            if key1[0] == key2[1] and key1[1] == key2[0]:
                key = value1 + value2
                link_degree_IML[key1] = key
                link_degree_IML[key2] = key

    for key1, value1 in link_degree_flow_common.items():
        for key2, value2 in link_degree_flow_common.items():
            if key1[0] == key2[1] and key1[1] == key2[0]:
                key = value1 + value2
                link_degree_flow_common[key1] = key
                link_degree_flow_common[key2] = key

    link_degree_flow_IML = link_degree_flow_common.copy()

    max_flow_common = link_degree_flow_IML[(1, 30)]
    min_flow_common = link_degree_flow_IML[(1, 30)]
    for key, value in link_degree_flow_IML.items():
        if max_flow_common < value:
            max_flow_common = value
        if min_flow_common > value:
            min_flow_common = value




    m1 = [[0 for i in range(50)] for i in range(50)]
    for key in link_degree:
        i = key[0][0] - 1
        j = key[0][1] - 1
        m1[i][j] = 1
        m1[j][i] = 1
        if key[1] == 1:
            m1[i][j] = 2
            m1[j][i] = 2
        elif key[1] == 2:
            m1[i][j] = 3
            m1[j][i] = 3
        elif key[1] == 3:
            m1[i][j] = 4
            m1[j][i] = 4


    use_ratio_GNT_C = []
    length_GNT_C = []
    path_use_GNT_C = []
    all_use_GNT_C = []
    change_ratio_GNT_C = []
    use_GNT_C = 0
    link_allflow_common_GNT_C = link_degree_flow_common.copy()
    network = m1
    p_gnt = []
    p_jie = []
    for key in link:
        p_gnt.append(key)
        for key1 in link_degree:
            if (key[0] == key1[0][0] and key[1] == key1[0][1]) or (key[1] == key1[0][0] and key[0] == key1[0][1]):
                p_jie.append(key1[1])
    link_degree_GNT = dict(zip(p_gnt, p_jie))
    tar_GNT = []

    a_obj = 0.3
    b_obj = 0.7

    for key, value in link_allflow_common_GNT_C.items():
        if link_degree_GNT[key] == 2:
            path_use_GNT_C = k_shortest_path(network, key[0], key[1], 7)
            all_use_GNT_C = 0
            obj = 100
            c2 = 0
            ratio = 0
            backup_path = []
            obj_list = []
            len_backup_path = 0

            change_GNT_C = 0
            all_change_GNT_C = 0
            change_ratio = 0
            for obj_key in path_use_GNT_C[1:]:
                i = 0
                all_use_GNT_C = 0

                for key1 in range(0, len(obj_key[0]) - 1):
                    jie = (obj_key[0][key1], obj_key[0][key1 + 1])
                    use_GNT_C = ((link_allflow_common_GNT_C[jie] + link_allflow_common_GNT_C[key])) / max_flow_common
                    all_use_GNT_C = all_use_GNT_C + use_GNT_C

                    r_use_GNT_C = link_allflow_common_GNT_C[jie] / max_flow_common
                    change_GNT_C = abs(use_GNT_C - r_use_GNT_C)
                    all_change_GNT_C = all_change_GNT_C + change_GNT_C
                    if use_GNT_C >= 1:
                        i = i + 1
                if i == 0:
                    c2 = a_obj * obj_key[1] + b_obj * (len(obj_key[0]) - 1)

                    obj_list.append(c2)
                    if obj > c2:
                        obj = c2
                        backup_path = obj_key[0].copy()
                        len_backup_path = len(obj_key[0]) - 1
                        ratio = all_use_GNT_C / len_backup_path
                        change_ratio = all_change_GNT_C / len_backup_path
            use_ratio_GNT_C.append(ratio)
            change_ratio_GNT_C.append(change_ratio)
            length_GNT_C.append(len_backup_path)
            network = m1
            tar_GNT.append(key)

        elif link_degree_GNT[key] == 3:
            path_use_GNT_C = k_shortest_path(network, key[0], key[1], 7)
            all_use_GNT_C = 0
            obj1 = 100
            obj2 = 100
            c3 = 0
            ratio1 = 0
            ratio2 = 0

            r_use_GNT_C = 0
            change_GNT_C = 0
            all_change_GNT_C = 0

            change_ratio1 = 0
            change_ratio2 = 0

            backup_path1 = []
            backup_path2 = []
            obj_list = []
            r = 0
            len_backup_path1 = 0
            len_backup_path2 = 0
            for obj_key in path_use_GNT_C[1:]:
                all_use_GNT_C = 0
                i = 0
                for key1 in range(0, len(path_use_GNT_C[1][0]) - 1):
                    jie = (path_use_GNT_C[1][0][key1], path_use_GNT_C[1][0][key1 + 1])
                    use_GNT_C = (link_allflow_common_GNT_C[jie] + link_allflow_common_GNT_C[
                        key] / 2) / max_flow_common
                    all_use_GNT_C = all_use_GNT_C + use_GNT_C

                    r_use_GNT_C = link_allflow_common_GNT_C[jie] / max_flow_common
                    change_GNT_C = abs(use_GNT_C - r_use_GNT_C)
                    all_change_GNT_C = all_change_GNT_C + change_GNT_C
                    if use_GNT_C >= 1:
                        i = i + 1
                if i == 0:
                    c3 = a_obj * obj_key[1] + b_obj * (len(obj_key[0]) - 1)

                    obj_list.append(c3)
                    if obj1 > c3:
                        obj1 = c3
                        backup_path1 = obj_key[0].copy()
                        len_backup_path1 = len(obj_key[0]) - 1
                        ratio1 = all_use_GNT_C / len_backup_path1
                        change_ratio1 = all_change_GNT_C / len_backup_path1
                    if obj1 < c3 and obj2 > c3:
                        obj2 = c3

                        backup_path2 = obj_key[0].copy()
                        for i in backup_path1[1:len(backup_path1) - 1]:
                            for j in backup_path2[1:len(backup_path2) - 1]:
                                if i == j:
                                    r = r + 1
                        if r == 0:
                            backup_path2 = obj_key[0].copy()
                            len_backup_path2 = len(obj_key[0]) - 1
                            ratio2 = all_use_GNT_C / len_backup_path2
                            change_ratio2 = all_change_GNT_C / len_backup_path1
            use_ratio_GNT_C.append((ratio1 + ratio2) / 2)
            change_ratio_GNT_C.append((change_ratio1 + change_ratio2) / 2)
            length_GNT_C.append(len_backup_path1 + len_backup_path2)
            network = m1
            tar_GNT.append(key)
        elif link_degree_GNT[key] == 1:
            ratio = 0
            ratio = link_allflow_common_GNT_C[key] / max_flow_common
            length_GNT_C.append(0)
            use_ratio_GNT_C.append(0)
            change_ratio_GNT_C.append(ratio)
            tar_GNT.append(key)
    print()
    use_ratio_GNT_C = dict(zip(tar_GNT, use_ratio_GNT_C))
    back_up_length_GNT = dict(zip(tar_GNT, length_GNT_C))
    change2 = dict(zip(tar_GNT, change_ratio_GNT_C))

    use_ratio_GNT_C1 = []
    for key, value in use_ratio_GNT_C.items():
        for key1, value1 in weights.items():
            if key1[0] == key[0] and key1[1] == key[1]:
                use_ratio_GNT_C1.append(value * 100)

    change_list2 = []
    for key, value in change2.items():
        for key1, value1 in weights.items():
            if key1[0] == key[0] and key1[1] == key[1]:
                change_list2.append(value * 100)

    len_all5 = []
    for key, value in back_up_length_GNT.items():
        for key1, value1 in weights.items():
            if key1[0] == key[0] and key1[1] == key[1]:
                len_all5.append(value)
    print("len_all5")
    print(len(len_all5))

    print("max_flow_common")
    print(max_flow_common)




    link_2 = []
    for i in range(1, 87):
        link_2.append(i)
    ratio_1 = []
    ratio_2 = []
    ratio_3 = []
    for key in range(0, 86):
        ratio_1.append(use_ratio_GNT_C1[key])
        ratio_2.append(use_ratio_IML1[key])
        ratio_3.append(use_ratio_d1[key])
    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(link_2, ratio_1, c='red', label="LSNC")
    plt.plot(link_2, ratio_2, c='green', linestyle='--', label="IML")
    plt.plot(link_2, ratio_3, c='blue', linestyle='-.', label="CR")
    plt.scatter(link_2, ratio_1, c='red')
    plt.scatter(link_2, ratio_2, c='green')
    plt.scatter(link_2, ratio_3, c='blue')
    plt.legend(loc='upper right', prop={'size': 19})
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(range(0, 100, 10), fontproperties='Times New Roman', size=20)
    plt.grid(True, linestyle='--', alpha=1)
    plt.xlabel("The number of links", fontdict={'size': 25})
    plt.ylabel("average bandwidth utilization rate", fontdict={'size': 25})
    plt.show()

    link_3 = []

    for i in range(1, 89):
        link_3.append(i)

    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(link_3, change_list2, c='red', label="LSNC")
    plt.scatter(link_3, change_list2, c='red')
    plt.legend(loc='upper right', prop={'size': 20})
    plt.xticks(fontsize=20, fontproperties='Times New Roman', size=20)
    plt.yticks(range(0, 100, 10), fontsize=20, fontproperties='Times New Roman', size=20)
    plt.grid(True, linestyle='--', alpha=1)
    plt.xlabel("The number of links", fontdict={'size': 25})
    plt.ylabel("average link utilization rate variation", fontdict={'size': 25})
    plt.show()


    link_1 = []
    for i in range(1, 88):
        link_1.append(i)
    plt.figure(figsize=(16, 9), dpi=100)
    len_all2_1 = []
    for key in range(0, 87):
        len_all2_1.append(len_all5[key])
    plt.plot(link_1, len_all2_1, c='red', label="LSNC")
    plt.legend(loc='upper right', prop={'size': 20})
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(range(0, 15, 5), fontsize=20)
    plt.grid(True, linestyle='--', alpha=2)
    plt.xlabel("The number of links", fontdict={'size': 25})
    plt.ylabel("backup path length", fontdict={'size': 25})
    plt.show()


