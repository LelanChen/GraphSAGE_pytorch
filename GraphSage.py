# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 14:21
# @Author  : chenlelan
# @File    : GraphSage.py

import torch.nn as nn
from SageGCN import SageGCN

'''
SAGE模型的作用是通过搭建多个单层网络来进行学习，实验中搭建的是一个两层网络。
'''
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        '''
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度列表，
            num_neighbors_list： 采样邻居数列表[一阶采样个数，二阶采样个数，...]
            num_layers: 网络深度，等于聚合邻域的跳数
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 1):     # 注意是len(hidden_dim) - 2而不是len(hidden_dim) - 1
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        # self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))   # hidden_dim[-1]表示列表最后一项
        self.fc_layer = nn.Linear(in_features=hidden_dim[-2], out_features=hidden_dim[-1])

    def forward(self, node_features_list):
        '''
        :param node_features_list: 节点特征列表，【源节点特征列表，一阶采样邻居特征列表，二阶采样邻居列表】
        :return:
        '''
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        final_hidden = hidden[0]
        logist = self.fc_layer(final_hidden)
        return logist

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )