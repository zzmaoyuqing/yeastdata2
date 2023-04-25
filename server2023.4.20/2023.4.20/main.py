import model
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import Data
from torch.utils.data import DataLoader
from args import args_parser

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.enabled = False

def main():
    args = args_parser()
    torch.manual_seed(args.seed)


    dataset = Data.MyDataset()
    train_data, val_data, test_data = Data.TensorDataset(dataset)

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    print("训练集的长度:{}".format(len(train_data)))
    print("测试集的长度:{}".format(len(val_data)))


    net = model.CNN(args)
    if torch.cuda.is_available():
        net = net.to(args.device)

    loss_func, optimizer = model.compile(args, net)


    model.train(args=args, model=net, train_loader=train_loader, loss_func=loss_func, optimizer=optimizer, val_loader=val_loader, epochs=args.epochs)



if __name__ == '__main__':
    main()