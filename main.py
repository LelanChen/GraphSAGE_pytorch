# -*- coding: utf-8 -*-
# @Time    : 2024/1/16 15:20
# @Author  : chenlelan
# @File    : main.py

'''
在Bot-Iot上的测试
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from GraphSage import GraphSage
from load_data import CoraData
from samplings import multihop_sampling, dynamic_sampling, get_sampling_index
from collections import namedtuple
from write_to_csv import write_eval_result
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score

INPUT_DIM = 12    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [24, 8, 2]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [3, 3]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST) + 1
BATCH_SIZE = 32     # 批处理大小
EPOCHS = 50
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.001    # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
#                            'train_mask', 'val_mask', 'test_mask'])
Data = namedtuple('Data', ['node_x', 'node_y', 'adjacency',
                           'test_x', 'test_y', 'test_adj'])

data = CoraData().data
x = data.node_x.iloc[:, 2:14].values / data.node_x.iloc[:, 2:14].values.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
train_index = np.arange(data.node_y.shape[0])
train_label = data.node_y
t_x = data.test_x.iloc[:, 2:14].values/data.test_x.iloc[:, 2:14].values.sum(1, keepdims=True) # 归一化测试数据
test_index = np.arange(data.test_y.shape[0])
test_label = data.test_y

model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
# print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


def train():
    model.train()
    for e in range(EPOCHS):
        print('*'*15, 'epochs ', e+1 , ' result', '*'*15)
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BATCH_SIZE,))
            # batch_src_label = torch.from_numpy(train_label[batch_src_index]).float().to(DEVICE)
            src_nodes = data.node_x.iloc[batch_src_index, 0:2]
            batch_sampling_result = dynamic_sampling(src_nodes, NUM_NEIGHBORS_LIST, data.adjacency)
            batch_sampling_index = get_sampling_index(batch_sampling_result, data.node_x)
            batch_label = []
            batch_logits = []
            for t in range(len(batch_sampling_index)):
                batch_src_label = torch.from_numpy(train_label[batch_sampling_index[t][0]])
                batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_index[t]]
                batch_train_logits = model(batch_sampling_x)
                # print(batch_train_logits.shape, batch_src_label.shape)
                # print('预测结果',batch_train_logits)
                # print('实际值',batch_src_label)
                batch_logits.append(batch_train_logits)
                batch_label.extend(batch_src_label)
            logits = torch.cat(batch_logits, dim=0)
            label = torch.Tensor(batch_label)
            # print(torch.cat(batch_logits, dim=0).shape, torch.Tensor(batch_label).shape)
            loss = criterion(logits, label.to(torch.long))
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            # exit()
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()
        today = datetime.today()
        save_path = 'result/GrapeSAGE_test_result_%s_%s_%s.pth' % (str(today.year),
                                              str(today.month), str(today.day))
        torch.save(model.state_dict(), save_path)


def test():
    print('*' * 15, 'test begin', '*' * 15)
    model.eval()
    batch_num = 1
    result = []
    with torch.no_grad():
        while batch_num * BATCH_SIZE < data.test_x.shape[0]:
            src_nodes = data.test_x.iloc[(batch_num-1) * BATCH_SIZE:batch_num * BATCH_SIZE, 0:2]
        # src_nodes = data.test_x.iloc[:, 0:2]
            test_sampling_result = dynamic_sampling(src_nodes, NUM_NEIGHBORS_LIST, data.test_adj)
            test_sampling_index = get_sampling_index(test_sampling_result, data.test_x)
            y_list = []
            for t in range(len(test_sampling_index)):
                test_node_x = [torch.from_numpy(t_x[idx]).float().to(DEVICE) for idx in test_sampling_index[t]]
                test_logits = model(test_node_x)
                y_list.append(test_logits)
            y = torch.cat(y_list, 0) # 在纵向进行拼接
            test_l = torch.from_numpy(test_label[(batch_num-1) * BATCH_SIZE:batch_num * BATCH_SIZE]).float().to(DEVICE)
            predict_y = y.max(-1)[1] # max(-1)中的-1表示按照最后一个维度（行）求最大值，方括号[1]则表示返回最大值的索引
            # print('test_label ’s dim：', test_l.shape)
            # accuarcy = torch.eq(predict_y, test_l).float().mean().item()
            test_result = get_result(test_l, predict_y)
            batch_num = batch_num + 1
        # result.append(test_result)
        # test_result = np.mean(np.array(result), axis=0)
        print("Test Accuracy: ", test_result[0])
        print("Test recall: ", test_result[1])
        print("Test precision: ", test_result[2])
        print("Test AUC: ", test_result[3])
        print("Test f1-score: ", test_result[4])
        today = datetime.today()
        output_file = 'result/GrapeSAGE_test_result_%s_%s_%s.csv'% (str(today.year),
                                              str(today.month), str(today.day))
        write_eval_result(test_result, output_file, 'GrapeSAGE', "Bot-Iot", mod='test')

def get_result(label, y_pred):
    # 计算准确率
    accuracy = accuracy_score(label, y_pred)
    # 计算召回率
    recall = recall_score(label, y_pred)
    # 计算精度
    precision = precision_score(label, y_pred)
    # AUC
    fpr, tpr, thresholds = roc_curve(label, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    # 计算 F1-score
    f1 = f1_score(label, y_pred)

    # print("Accuracy:", accuracy)
    # print("Recall:", recall)
    # print("Precision:", precision)
    # print("AUC:", AUC)
    # print("F1-score:", f1)
    result_scores = [accuracy, recall, precision, AUC, f1]
    return result_scores

if __name__ == '__main__':
    train()