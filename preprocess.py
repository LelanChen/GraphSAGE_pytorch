# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 17:12
# @Author  : chenlelan
# @File    : preprocess.py

import os.path as osp
import pickle
import numpy as np
import pandas as pd
import itertools
import scipy.sparse as sp
import urllib
from collections import namedtuple
import os


# namedtuple能够用来创建类似于元组的数据类型，除了能够用索引来访问数据，能够迭代，还能够方便的通过属性名来访问数据。
# Point = namedtuple('Point',['x','y']) # 类名为Point,属性有'x'和'y'
"""
pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
pickle.load(file)函数的功能：将file中的对象序列化读出。
pickle.dumps(obj[, protocol])函数的功能：将obj对象序列化为string形式，而不是存入文件中。
pickle.loads(string)函数的功能：从string中读出序列化前的obj对象。
"""

# Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])
Data = namedtuple('Data', ['node_x', 'node_y', 'adjacency', 'test_x', 'test_y', 'test_adj'])
class CoraData(object):
    filenames = ['iscx-test-{}'.format(name) for name in ['node-feature_1.csv', 'adj-dict.pickle', '1000-node-feature_1.csv', '1000-adj-dict.pickle']]
    def __init__(self, data_root='./data/cora2' , rebuild=False):
        """
        读取数据以及建图
        :param data_root:数据存储路径
        :param rebuild: 是否需要重建数据，否的时候将其设置为False，是的时候将其设为True
        x:节点的特征,维度为2708*1433，类型为 np.ndarray
        y:节点的标签，总共包括7个类别，类型为 np.ndarray
        adjacebcy_dict: 邻接信息，类型为dict
        train_mask: 训练集掩码向量，维度为2708，当节点属于训练集时，相应的位置为True，否则为False
        val_mask: 验证集掩码向量，维度为2708，当节点属于验证集时，相应位置为True，否则为False
        test_mask: 测试集掩码向量，维度为2708，当节点属于测试集时，相应位置为True，否则为False
        """
        self.data_root = data_root
        # pkl文件是python里面保存文件的一种格式，如果直接打开会显示一堆序列化的东西（二进制文件）。
        # 常用于保存神经网络训练的模型或者各种需要存储的数据。
        # save_file = "D:/code/python/dp/GNN/SAGE/cora1/cora1/cached.pkl"
        save_file = osp.join(self.data_root, "iscx.pkl")
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
        tx(numpy.ndarray)：(1000, 1433)
        allx(numpy.ndarray):(1708, 1433)
        y(numpy.ndarray):(140, 7)
        ty(numpy.ndarray):(1000, 7)
        ally(numpy.ndarray):(1708, 7)
        graph(defaultdict(<class list>)):ex{0:[633, 1862, 2582]}    # 邻接表（字典类型），key是src节点，value是指向的dst节点列表

        :return:
        """
        print('Process data ...')
        node_df, adj_list, test_df, test_adj = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        node_x = node_df.iloc[:, 0:11]  # 读取指定列（不包括第11列）获取节点特征
        node_y = node_df.loc[:, 'Label'].values
        test_x = test_df.iloc[:, 0:11]
        test_y = test_df.loc[:, 'Label'].values
        # num_nodes = node_x.shape[0]

        print("Node's feature shape:", node_x.shape)
        print("Node's label shape:", node_y.shape)
        print("Adjacency's shape:", len(adj_list))
        # print("NUmber of training nodes:", train_mask.sum())
        # print("Number of validation nodes:", val_mask.sum())
        # print("Number of test nodes:", test_mask.sum())

        return Data(node_x=node_x, node_y=node_y, adjacency=adj_list, test_x=test_x, test_y=test_y, test_adj=test_adj)


    @staticmethod
    def read_data(path):
        """
        使用不同的方式读取原始数据进一步处理
        :param path:
        :return:
        """
        name = osp.basename(path)
        if 'adj-dict' in name:
            if '.csv' in name:
                # 将每行列数不等长的csv文件转换成dataframe
                largest_colum = 0
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        colum_count = len(l.split(' '))+1
                        # 找到列数最多的行
                        largest_colum = colum_count if largest_colum < colum_count else largest_colum
                f.close()
                print('最大列数：',  largest_colum)
                out = pd.read_csv(path, header=None, sep=' ', names=range(largest_colum))
                # 填补缺失值
                out = out.fillna('')
            if '.pickle' in name:
                f = open(path, 'rb')
                out = pickle.load(f)
        else:
            out = pd.read_csv(path)
        print(out)
        return out

# 测试实列
# data = CoraData().data