import os.path
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import wandb
from argparse import Namespace
import math

# 构建模型思路： 输入 -> 节点嵌入 -> 卷积 -> Max pooling -> 展平 -> sigmoid分类 -> 输出
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # 32, 32, 26, 26
            nn.MaxPool2d((2, 2)),      # 32, 32, 13, 13
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),  # 32, 64, 11, 11
            nn.Dropout(0.5),
            nn.MaxPool2d((2, 2))        # 32, 64, 5, 5
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3)),  # 32, 128, 3, 3
            nn.Dropout(0.5),
            nn.MaxPool2d((2, 2))        # 32, 128, 1, 1
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),

            nn.ReLU(),
            nn.Linear(128, 32),

            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # # init cnn model
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def forward(self, input):
        input = input.to(torch.float32)
        input = input.permute(1, 0, 2, 3)  # ([32, 1, 28, 28])   B,I,W,H

        input = self.conv1(input)  # ([32, 32, 13, 13])  B,I,W,H
        input = self.conv2(input)  # ([32, 64, 5, 5])
        input = self.conv3(input)  # ([32, 128, 1, 1])

        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        input = input.view(-1, 1 * 1 * 128)  # input.shape:([32, 128])    x.view的第二个参数和nn.Linear第一个参数一致
        output = self.decoder(input)
        # print('output.shape', output.shape)  # torch.Size([32, 2])

        return output


# 定义损失函数、优化器
def compile(args, model):
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer.zero_grad()
    return loss_func, optimizer


def compute_metrics(all_trues, all_scores, threshold=0.5):
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    p, r, _ = metrics.precision_recall_curve(all_trues, all_scores)
    AUPR = metrics.auc(r, p)

    return acc, f1, pre, rec, mcc, AUC, AUPR


def train_one_epoch(args, model, train_loader, loss_func, optimizer, wandb):
    model.train()

    total_train_step = 0  # 记录训练的次数
    total_train_loss = 0
    start_time = time.time()

    all_trues = []
    all_scores = []

    # 训练模型
    for idx, data in enumerate(train_loader):
        X, y = data
        # print('X的shape：', X.shape)   # torch.Size([32, 28, 28])
        # print('y的shape:', y.shape)   # torch.Size([32, 1])
        X = X.unsqueeze(0)
        # print('X的shape：unsqueeze', X.shape)  # torch.Size([1, 32, 28, 28])
        y = y.long()
        y = y.squeeze(1)
        # print('y.squeeze的shape：', y.shape)  # torch.Size([32])
        if torch.cuda.is_available():
            X = X.to(args.device)
            y = y.to(args.device)

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        total_train_loss += loss.item()  # 记录一轮的loss总值

        # 优化器
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        total_train_step += 1
        wandb.log({
            'train_one_epoch loss': loss
        })

        # 每一个epoch：隔24个iteration输出一次
        if total_train_step % 24 == 0:
            end_time = time.time()
            print("运行时间：", end_time - start_time)
            print("第{}次训练的loss={}  ".format(total_train_step, loss.item()))
        all_trues.append(y.data.cpu().numpy())
        all_scores.append(y_pred.argmax(1).data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores)
    return total_train_loss, acc, f1, pre, rec, mcc, AUC, AUPR


def val_one_epoch(args, model, val_loader, loss_func):
    model.eval()

    total_val_step = 0  # 记录测试的次数
    total_val_loss = 0
    all_trues = []
    all_scores = []
    with torch.no_grad():
        for data in val_loader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有32个数据。
            X, y = data
            X = X.unsqueeze(0)
            y = y.long()
            y = y.squeeze(1)  # torch.Size([32,])
            if torch.cuda.is_available():
                X = X.to(args.device)
                y = y.to(args.device)

            y_pred = model(X)
            loss = loss_func(y_pred, y)
            total_val_loss += loss.item()
            # print('测试集预测的概率：', y_pred)

            all_trues.append(y.data.cpu().numpy())
            all_scores.append(y_pred.argmax(1).data.cpu().numpy())
        all_trues = np.concatenate(all_trues, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        # es = all_scores.sum()
        # print('test')
        acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores)
    return total_val_loss, acc, f1, pre, rec, mcc, AUC, AUPR


def train(args, model, train_loader, loss_func, optimizer, val_loader, epochs):
    # 配置wandb
    wandb_config = Namespace(
        project_name='3conv2linear',
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay
    )
    # 初始化wandb
    wandb.init(
        project=wandb_config.project_name,
        config=wandb_config.__dict__,
    )
    best_acc = 0.0

    # 训练模型
    for epoch in range(epochs):
        print("------------------------第{}轮训练------------------------".format(epoch + 1))
        train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_one_epoch(
            args=args,
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            wandb=wandb
        )
        wandb.log({
            'train_epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_rec': train_rec,
            'train_mcc': train_mcc,
            'train_auc': train_auc,
            'train_aupr': train_aupr
        })

        val_loss, val_acc, val_f1, val_pre, val_rec, val_mcc, val_auc, val_aupr = val_one_epoch(
            args=args,
            model=model,
            val_loader=val_loader,
            loss_func=loss_func
        )

        wandb.log({
            'val_epoch': epoch + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_rec': val_rec,
            'val_mcc': val_mcc,
            'val_auc': val_auc,
            'val_aupr': val_aupr
        })

        res = '\t'.join([
            '\nEpoch[%d/%d]' % (epoch + 1, epochs),
            '\nTrain dataset',
            'loss:%0.5f' % train_loss,
            'accuracy:%0.6f' % train_acc,
            'f-score:%0.6f' % train_f1,
            'precision:%0.6f' % train_pre,
            'recall:%0.6f' % train_rec,
            'mcc:%0.6f' % train_mcc,
            'auc:%0.6f' % train_auc,
            'aupr:%0.6f' % train_aupr,
            '\nVal dataset',
            'loss:%0.5f' % val_loss,
            'accuracy:%0.6f' % val_acc,
            'f-score:%0.6f' % val_f1,
            'precision:%0.6f' % val_pre,
            'recall:%0.6f' % val_rec,
            'mcc:%0.6f' % val_mcc,
            'auc:%0.6f' % val_auc,
            'aupr:%0.6f' % val_aupr,

        ])
        print(res)

        if val_acc > best_acc:
            best_acc = val_acc
            print("save model{}".format(epoch + 1))
            torch.save(model.state_dict(), 'model_pytorch_test/model_{}.pth'.format(epoch + 1))
            # torch.save(model, 'model_pytorch_test/model_{}.pth'.format(epoch + 1))
        # print("第{}轮模型训练数据已保存".format(epoch + 1))

    wandb.finish()