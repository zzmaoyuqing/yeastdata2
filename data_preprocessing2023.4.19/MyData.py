import pandas as pd
import torch as t
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data as Data

import torch
from torch.utils.data import random_split


seed = 42
torch.manual_seed(seed)

class MyDataset(Dataset):
    """ my dataset: ensure the data and target connected

    input file : concat_emb_sub.npy    ndarray:(5093,2,128)
                 protein_target.txt
                            x_data     ndarray:(5093,2,128)
                            y_data     ndarray:(5093,)
                            protein    ndarray:(5093,)
    """

    def __init__(self):
        # 读取csv文件中的数据
        X = np.load('data/dot_emb_sub.npy')   #  PPI的embedding处理为（0，1）范围
        self.len = X.shape[0]
        # 最后一列为标签为，存在y_data中
        self.protein_target = pd.read_csv('node2vec/data/protein_target.txt')   # protein_target.txt的第一行当表头
        self.protein = self.protein_target.iloc[:, 1].values
        self.x_data = X
        self.y_data = self.protein_target.iloc[:, 2].values

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index], self.protein[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

# 划分数据集: 都是subset数据类型
def split_data(dataset):
    torch.manual_seed(seed)
    train_dataset, test_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[0.6, 0.2, 0.2])
    return train_dataset, test_dataset, val_dataset

# 读取划分后的数据
def read_split_data(dataset, splited_dataset):
    '''
    :param dataset: 读入的concat_emb_sub.npy数据
    :param splited_dataset: 划分后的数据
    :return:
    '''
    indices = splited_dataset.indices
    splited_data = []
    for i in range(0, len(indices)):
        a = dataset.x_data[indices[i]]
        splited_data.append(a)
    return splited_data

# 读取划分后数据的target
def read_split_target(dataset, splited_dataset):
    indices = splited_dataset.indices
    splited_target = np.zeros((1, len(indices)))
    for i in range(0, len(indices)):
        splited_target[0, i] = dataset.y_data[indices[i]]
    return splited_target

if __name__ == '__main__':
    # 读入数据
    dataset = MyDataset()
    # 划分数据 0.6、0.2、0.2
    train_dataset, test_dataset, val_dataset = split_data(dataset)  # subset类
    # 分别读取划分后的train、val、test数据
    train_data = read_split_data(dataset, train_dataset)
    val_data = read_split_data(dataset, val_dataset)
    test_data = read_split_data(dataset, test_dataset)

    # 分别读取划分后的train_target, val_target、 test_target标签
    train_target = read_split_target(dataset, train_dataset)
    # train_target = torch.stack(train_target)  # list中都是tensor型向量，最终将list整体转成tensor
    val_target = read_split_target(dataset, val_dataset)
    test_target = read_split_target(dataset, test_dataset)

    # 记录各个集合中关键蛋白的数量
    train_es_num = train_target.sum()  # 711   train_data=3056
    val_es_num = val_target.sum()      # 241   val_data=1018
    test_es_num = test_target.sum()    # 215   test_data=1019

    np.save('data/train_set/train_data.npy', train_data)
    np.save('data/train_set/train_target.npy', train_target)

    np.save('data/val_set/val_data.npy', val_data)
    np.save('data/val_set/val_target.npy', val_target)

    np.save('data/test_set/test_data.npy', test_data)
    np.save('data/test_set/test_target.npy', test_target)


