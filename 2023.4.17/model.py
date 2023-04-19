import os.path
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import wandb
from argparse import Namespace
# 构建模型思路： 输入 -> 节点嵌入 -> 卷积 -> Max pooling -> 展平 -> sigmoid分类 -> 输出
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      kernel_size=args.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.kernel_size))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64*63*1, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )




    def forward(self, input):
        input = input.to(torch.float32)
        # input = input.permute(0, 2, 1)
        output = self.fc(self.conv(input))

        return output


# 定义损失函数、优化器
def compile(args, model):
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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


def train_one_epoch(args, model, train_loader, loss_func, optimizer):
    model.train()

    total_train_step = 0     # 记录训练的次数
    total_train_loss = 0
    start_time = time.time()

    all_trues=[]
    all_scores=[]

    # 训练模型
    for idx, data in enumerate(train_loader):
        X, y = data
        y = y.long()
        y = y.squeeze(1)
        if torch.cuda.is_available():
            X = X.to(args.device)
            y = y.to(args.device)

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        total_train_loss += loss.item() # 记录一轮的loss总值

        # 优化器
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        # # 将训练集的loss写入tensorboard
        # writer.add_scalar('Loss/train', loss.item(), total_train_step)
        # writer.add_scalar('Accuracy/train', )

        total_train_step += 1

        # 每一个epoch：隔24个iteration输出一次
        if total_train_step % 24 == 0:
            end_time = time.time()
            print("运行时间：", end_time - start_time)
            print("第{}次训练的loss={}  ".format(total_train_step, loss.item()))
        all_trues.append(y.data.numpy())
        all_scores.append(y_pred.argmax(1).data.numpy())

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
            y = y.long()
            y = y.squeeze(1)  # torch.Size([32,])
            if torch.cuda.is_available():
                X = X.to(args.device)
                y = y.to(args.device)

            y_pred = model(X)
            loss = loss_func(y_pred, y)
            total_val_loss += loss.item()
            # print('测试集预测的概率：', y_pred)

            all_trues.append(y.data.numpy())
            all_scores.append(y_pred.argmax(1).data.numpy())
        all_trues = np.concatenate(all_trues, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores)
    return total_val_loss, acc, f1, pre, rec, mcc, AUC, AUPR


def train(args, model, train_loader, loss_func, optimizer,  val_loader, epochs):
    # 配置wandb
    wandb_config = Namespace(
        project_name='1conv4linear',
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    # 初始化wandb
    wandb.init(
        project=wandb_config.project_name,
        config=wandb_config.__dict__,
    )

    # 训练模型
    for epoch in range(epochs):
        print("------------------------第{}轮训练------------------------".format(epoch+1))
        train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_one_epoch(
            args=args,
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
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
            'aupr:%0.6f'% train_aupr,
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

        torch.save(model, 'model_pytorch_test/model_{}.pth'.format(epoch + 1))
        print("第{}轮模型训练数据已保存".format(epoch + 1))
    wandb.finish()