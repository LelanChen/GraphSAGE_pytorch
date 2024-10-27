# -*- coding: utf-8 -*-
# @Time    : 2024/6/8 19:49
# @Author  : chenlelan
# @File    : load_data.py

import os.path as osp
import pickle
import pandas as pd
from collections import namedtuple
import numpy as np
import networkx as nx
from collections import namedtuple
import os


# namedtuple能够用来创建类似于元组的数据类型，除了能够用索引来访问数据，能够迭代，还能够方便的通过属性名来访问数据。
# Point = namedtuple('Point',['x','y']) # 类名为Point,属性有'x'和'y'

Data = namedtuple('Data', ['node_x', 'node_y', 'adjacency', 'test_x', 'test_y', 'test_adj'])
class CoraData(object):
    filenames = ['node_features_without_new.csv', 'graphs_remap.pkl', 'node_features_without_new.csv', 'graphs_remap.pkl']
    def __init__(self, data_root='./data/Bot-Iot' , rebuild=False):
        """
        读取数据以及建图
        :param data_root:数据存储路径
        :param rebuild: 是否需要重建数据，否的时候将其设置为False，是的时候将其设为True
        x:节点的特征,维度为2708*1433，类型为 np.ndarray
        y:节点的标签，总共包括7个类别，类型为 np.ndarray
        adjacebcy_dict: 邻接信息，类型为dict
        """
        self.data_root = data_root
        # pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西（二进制文件）。
        # 常用于保存神经网络训练的模型或者各种需要存储的数据。
        # save_file = "D:/code/python/dp/GNN/SAGE/cora1/cora1/cached.pkl"
        save_file = osp.join(self.data_root, "data.pkl")
        print(save_file)
        if osp.exists(save_file) and not rebuild:
            print('Using Cached file:{}'.format(save_file))
            self._data = pickle.load(open(save_file, 'rb'))
        else:
            self._data = self.process_data()
            with open(save_file, 'wb') as f:
                pickle.dump(self.data, f)
            print('Cached file:{}'.format(save_file))

    @property
    # @property可以使方法像属性一样被调用
    def data(self):
        """
        返回Data数据类型,包括x,y,adjacency,train_mask,val_mask,test_mask
        :return:
        """
        return self._data

    def process_data(self):
        """
        处理数据，得到节点的特征和标签，邻接矩阵，训练集，验证集以及测试集
        :return:
        """
        print('Process data ...')
        node_df, adj_list, test_df, test_adj = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        num_column = node_df.shape[1]
        node_x = node_df.iloc[:, 0: num_column-2]  # 读取指定列（不包括最后两列label和classes）获取节点特征
        node_y = node_df.loc[:, 'Label'].values
        test_x = test_df.iloc[:, 0:num_column-2]
        test_y = test_df.loc[:, 'Label'].values
        # num_nodes = node_x.shape[0]

        # 将邻接矩阵转换成邻接字典{src: dest...}
        train_src_list = node_df.iloc[:, 0: 2]
        train_adj_dict = self.get_adj_dict(adj_list, train_src_list)
        test_src_list = test_df.iloc[:, 0: 2]
        test_adj_dict = self.get_adj_dict(test_adj, test_src_list)

        print("Node's feature shape:", node_x.shape)
        print("Node's label shape:", node_y.shape)
        print("Adjacency's shape:", len(adj_list))
        # print("NUmber of training nodes:", train_mask.sum())
        # print("Number of validation nodes:", val_mask.sum())
        # print("Number of test nodes:", test_mask.sum())

        return Data(node_x=node_x, node_y=node_y, adjacency=train_adj_dict, test_x=test_x, test_y=test_y, test_adj=test_adj_dict)

    def get_adj_dict(self, adj_list, src_list):
        adj_dict_list = []
        for t in range(len(adj_list)):
            adj_dict = {}
            adj_matrix = adj_list[t]
            src_t = src_list[src_list["Time"] == t].iloc[:, -1]
            print("{}时刻的原节点列表：{}".format(t, src_t))
            for i in range(adj_matrix.shape[0]):
                non_zero_indices = np.nonzero(adj_matrix[i])[0]
                adj_dict.update({src_t.to_list()[i]: src_t.iloc[non_zero_indices].to_list()})
            adj_dict_list.append(adj_dict)
            # print(adj_dict_list)
        return adj_dict_list

    @staticmethod
    def read_data(path):
        """
        使用不同的方式读取原始数据进一步处理
        :param path:
        :return:
        """
        name = osp.basename(path)
        graphs = None
        if 'graphs' in name:
            if '.npz' in name:
                graphs = np.load("data/Enron_new/graphs.npz", encoding='latin1', allow_pickle=True)['graph']
                print("Loaded {} graphs ".format(len(graphs)))

            if '.pkl' in name:
                with open('D:/科研/小论文/HADGA-pytorch/data/DDoS/graphs_remap.npz', 'rb') as f:
                    graphs = pickle.load(f)
            adj_matrices = list(map(lambda x: np.array(nx.adjacency_matrix(x).todense()), graphs))
            out = adj_matrices
        else:
            out = pd.read_csv(path)
        # print(out)
        return out


# 测试实列
# data = CoraData().data