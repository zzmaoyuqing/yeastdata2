import model
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import MyData
import balanced_sampler
from torch.utils.data import DataLoader
from args import args_parser
import random
import numpy as np

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.enabled = False

def main():
    args = args_parser()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # 设置随机种子
    setup_seed(42)
    # 加载train_data
    dataset = MyData.MyDataset()
    train_data, val_data, test_data = MyData.TensorDataset(dataset)
    # 平衡采样train_data
    print('{:-^40}'.format("train_data"))
    balanced_data, balanced_target = balanced_sampler.balanced_dataset(train_data)
    # 平衡采样val_data
    print('{:-^40}'.format("val_data"))
    val_balanced_data, val_balanced_target = balanced_sampler.balanced_dataset(val_data)

    balanced_sampler.list2tensor(balanced_data, balanced_target)
    balanced_sampler.list2tensor(val_balanced_data, val_balanced_target)

    bd = balanced_sampler.MyBalancedDataset(balanced_data, balanced_target, val_balanced_data, val_balanced_target)

    balanced_train_dataset1, balanced_val_dataset1, balanced_train_dataset2, balanced_val_dataset2, \
    balanced_train_dataset3, balanced_val_dataset3, balanced_train_dataset4, balanced_val_dataset4 = balanced_sampler.TensorDataset(bd)

    # loader data
    train_loader = DataLoader(balanced_train_dataset1, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(balanced_val_dataset1, batch_size=args.batch_size, shuffle=True)
    print("\n训练集的长度:{}".format(len(balanced_train_dataset1)))
    print('number of essential protein in balanced_train_dataset1:', int(balanced_train_dataset1.tensors[1].sum()))
    print("测试集的长度:{}".format(len(balanced_val_dataset1)))
    print('number of essential protein in balanced_val_dataset1:', int(balanced_val_dataset1.tensors[1].sum()))


    net = model.CNN(args)
    if torch.cuda.is_available():
        net = net.to(args.device)

    loss_func, optimizer = model.compile(args, net)


    model.train(args=args, model=net, train_loader=train_loader, loss_func=loss_func, optimizer=optimizer, val_loader=val_loader, epochs=args.epochs)



if __name__ == '__main__':
    main()