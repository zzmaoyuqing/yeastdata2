import random
import torch
import numpy as np

import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import MyData


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 平衡采样方法，换数据集的话手动修改一下balanced_data和balanced_target的初始化
def balanced_dataset(train_data):
    positive_indices = [i for i in range(len(train_data)) if train_data[i][1] == 1]
    negative_indices = [i for i in range(len(train_data)) if train_data[i][1] == 0]
    # 打乱negative_indices，方便后续取数据
    random.shuffle(negative_indices)
    # print(negative_indices)

    positive_num = np.array(train_data.tensors[1]).sum()
    negative_num = len(train_data)-positive_num
    print('positive:{:d}个'.format(int(positive_num)))   # 打印输出原始数据集样本分类分布
    print('negative:{:d}个'.format(int(negative_num)))

    # 每隔int(positive_num) 711个元素（最后一组是212个元素）取一组negative_indices的值。 ret每组的数加起来等于negative_num
    ret = []
    for i in range(0, len(negative_indices), int(positive_num)):
        ret.append(negative_indices[i:i+int(positive_num)])
        # print(negative_indices[i:i+int(positive_num)])


    # 将所有非关键蛋白2345个  每隔int(positive_num) 711个元素（最后一组是212个元素）取出negative_indices对应的train_data的值和标签
    # 711 711 711 212
    train_negative_data = []
    train_negative_target = []

    for i in range(0, len(ret)):
        if len(ret[i]) == int(positive_num):  # 711个元素
            for idx in ret[i]:
                train_negative_data.append(train_data.tensors[0][idx])
                train_negative_target.append(train_data.tensors[1][idx])
        else:  # 不是711个元素
            for idx in ret[-1]:
                train_negative_data.append(train_data.tensors[0][idx])
                train_negative_target.append(train_data.tensors[1][idx])

    # 所有的关键蛋白711个
    train_positive_data = list(train_data.tensors[0][positive_indices])
    train_positive_target = list(train_data.tensors[1][positive_indices])

    # 每隔int(positive_num) 711个元素（最后一组是212个元素）取一组train_negative_data和train_negative_target值。
    # res和tar是空list 不可以改
    res = []
    tar = []
    for i in range(0, len(negative_indices), int(positive_num)):
        res.append(train_negative_data[i:i+int(positive_num)])
        tar.append(train_negative_target[i:i+int(positive_num)])

    # 平衡采样，实现每组data都有711个关键蛋白和711个非关键蛋白；最后一组是212个关键蛋白，212个非关键蛋白
    balanced_data = [0]*len(ret)
    balanced_target = [0]*len(ret)
    for i in range(0, len(ret)):
        if len(ret[i]) == int(positive_num):  # 711个元素
            balanced_data[i] = res[i] + train_positive_data
            balanced_target[i] = tar[i] + train_positive_target
        else:  # 不是711个元素
            balanced_data[i] = res[-1] + train_positive_data[0:len(res[-1])]
            balanced_target[i] = tar[-1] + train_positive_target[0:len(res[-1])]
    return  balanced_data, balanced_target

# 将list--->tensor: torch.stack
def list2tensor(balanced_data, balanced_target):
    for i in range(len(balanced_data)):
        balanced_data[i] = torch.stack(balanced_data[i])
        balanced_target[i] = torch.stack(balanced_target[i])

# 将平衡采样的数据封装为Dataset
class MyBalancedDataset(Dataset):
    def __init__(self, balanced_data, balanced_target, val_balanced_data, val_balanced_target):
        self.train_data = balanced_data

        self.train_target = balanced_target
        self.train_len = len(self.train_data)

        self.val_data = val_balanced_data
        self.val_target = val_balanced_target
        self.val_len = len(self.val_data)

    def __getitem__(self, idx):
        # 根据索引返回数据和对应的标签
        return self.train_data[idx], self.train_target[idx], self.val_data[idx], self.val_target[idx]

    def __len__(self):
        # 返回文件数据的数目
        return self.train_len, self.val_len

def TensorDataset(dataset):
    train_x = dataset.train_data                            # list:4 里面每一个元素是tensor： torch.Size([1422, 28, 28])*3 + torch.Size([424, 28, 28])
    train_y = dataset.train_target                          # list:4 里面每一个元素是tensor： torch.Size([1422, 1])*3 + torch.Size([424, 1])

    val_x = dataset.val_data
    val_y = dataset.val_target

    balanced_train_dataset1 = Data.TensorDataset(train_x[0], train_y[0])
    balanced_val_dataset1 = Data.TensorDataset(val_x[0], val_y[0])

    balanced_train_dataset2 = Data.TensorDataset(train_x[1], train_y[1])
    balanced_val_dataset2 = Data.TensorDataset(val_x[1], val_y[1])

    balanced_train_dataset3 = Data.TensorDataset(train_x[2], train_y[2])
    balanced_val_dataset3 = Data.TensorDataset(val_x[2], val_y[2])

    balanced_train_dataset4 = Data.TensorDataset(train_x[3], train_y[3])
    balanced_val_dataset4 = Data.TensorDataset(val_x[3], val_y[3])

    return balanced_train_dataset1, balanced_val_dataset1, balanced_train_dataset2, balanced_val_dataset2, balanced_train_dataset3, balanced_val_dataset3, balanced_train_dataset4, balanced_val_dataset4
