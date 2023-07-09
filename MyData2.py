import numpy as np
import torch as t
import torch.utils.data as Data

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, train_data, train_target,test_data, test_target):
        # 2. import the dataset with trainset : testset = 0.8 : 0.2
        self.train_data = train_data
        self.train_target = train_target
        self.train_len = self.train_data.shape[0]

        self.test_data = test_data
        self.test_target = test_target
        self.test_len = self.test_data.shape[0]
    def __getitem__(self, idx):
        # 根据索引返回数据和对应的标签
        return self.train_data[idx], self.train_target[idx], self.test_data[idx], self.test_target[idx]

    def __len__(self):
        # 返回文件数据的数目
        return self.train_len, self.test_len

def TensorDataset(dataset):
        train_x = dataset.train_data  # ndarray(3056,28,28)
        train_x = t.tensor(train_x)  # Tensor(3056,28,28)
        train_y = dataset.train_target  # ndarray(1,3056)
        train_y = t.tensor(train_y.reshape(len(train_x), -1))  # ndarray(1,3056)--->Tensor(1,3056)---->Tensor(3056,1)

        test_x = dataset.test_data
        test_x = t.tensor(test_x)
        test_y = dataset.test_target
        test_y = t.tensor(test_y.reshape(len(test_x), -1))

        torch_train_dataset = Data.TensorDataset(train_x, train_y)
        torch_test_dataset = Data.TensorDataset(test_x, test_y)

        return torch_train_dataset, torch_test_dataset

if __name__ == '__main__':
    # 读入数据
    train_data = np.load('dataset/dim224/train_test/train_set/train_data.npy')

    train_target = np.load('dataset/dim224/train_test/train_set/train_target.npy')

    test_data = np.load('dataset/dim224/train_test/test_set/test_data.npy')
    test_target = np.load('dataset/dim224/train_test/test_set/test_target.npy')

    dataset = MyDataset(train_data, train_target, test_data, test_target)
    torch_train_dataset, torch_test_dataset = TensorDataset(dataset)

    # train_loader = DataLoader(torch_train_dataset, batch_size=4)
    # for data in train_loader:
    #     emb, target = data
    #     print(emb.shape)
    #     print(target)
