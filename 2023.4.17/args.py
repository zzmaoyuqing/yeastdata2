import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--m',)# model怎么换？

    parser.add_argument('--device', default='cuda', help='device id(i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--in_channels', type=int, default=2, help='in_channels')
    parser.add_argument('--out_channels', type=int, default=64, help='out_channels')
    parser.add_argument('--kernel_size', type=int, default=2, help='kernel size')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epoch')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=42,help='set seed')


    args = parser.parse_args()

    return args
