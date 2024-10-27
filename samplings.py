# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 10:42
# @Author  : chenlelan
# @File    : samplings.py

import numpy as np

'''单阶采样'''
def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表

    Returns:
        np.ndarray -- 采样结果构成的列表（返回的是节点ip）
    """
    results = []
    # print("src_nodes is:", src_nodes)
    for src in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        # print("src_nodes[sid] {} 's neighbor_table {}:".format(src, neighbor_table[src]))
        res = np.random.choice(neighbor_table[src], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()    # flatten()对多维数据进行压缩，压缩为1维数组


'''多阶采样'''
def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 中心节点
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射

    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result

def dynamic_sampling(src_nodes, sample_nums, neighbor_table):
    '''
    对每个时刻的静态图快照进行采样
    :param src_nodes:
    :param sample_nums:
    :param neighbor_table: list of dict
    :return:
    返回每个时刻的各阶采样结果列表【t1 [[0-adj] [1-adj] [2-adj]],...,tn [[0-adj] [1-adj] [2-adj]]】
    '''
    t_diff = src_nodes['Time'].value_counts().index.tolist()
    sample_result = []
    t_diff = sorted(t_diff)
    # print("时间序列", t_diff)
    for t in t_diff:
        # print('*'*10,'时间',t ,type(t),"*"*10)
        src_df = src_nodes[src_nodes['Time'] == t]
        src_n = src_df['Ip'].to_numpy()
        adj_dict = neighbor_table[t]
        # for src, dst in adj_dict.items():
        #     print("当前邻居列表",src, dst)
        sample_t = multihop_sampling(src_n, sample_nums, adj_dict)
        # print("{}时刻的采样列表{}".format(t, sample_t))
        sample_result.append(sample_t)
    return sample_result

def get_sampling_index(sample_result, node_x):
    n = np.array(sample_result).ndim # 采样结果的维度，1维是静态图采样结果，2维是时间图采样结果
    # print('采样列表维度', n)
    if n == 1:
        for src in sample_result:
            sample_index = node_x[(node_x['Ip'] == src)].index.tolist()
    else:
        sample_index = []
        for t in range(len(sample_result)):
            sample_t = []
            # print('时间', t)
            for i in range(len(sample_result[t])):
                sample_i = []
                for src in sample_result[t][i]:
                    index_t = node_x[(node_x['Time'] == t) & (node_x['Ip'] == src)].index.tolist()
                    sample_i.extend(index_t)
                sample_t.append(sample_i)
            # print('时间', t, '时的采样索引', sample_t)
            sample_index.append(sample_t)
    return sample_index




