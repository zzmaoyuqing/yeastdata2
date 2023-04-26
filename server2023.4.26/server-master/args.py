import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--m',)# model怎么换？

    parser.add_argument('--device', default='cuda', help='device id(i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--in_channels', type=int, default=1, help='in_channels')
    parser.add_argument('--out_channels', type=int, default=32, help='out_channels')
    parser.add_argument('--kernel_size', type=int, default=(3, 3), help='kernel size')

    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--seed', type=int, default=42, help='set seed')



    args = parser.parse_args()

    return args
