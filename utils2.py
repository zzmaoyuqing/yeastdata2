import torch
from sklearn import metrics
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from argparse import Namespace
# import wandb


import numpy as np
import torch
import os

# 定义损失函数、优化器
def compile(config, model):
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer.zero_grad()
    return loss_func, optimizer


def compute_metrics(all_trues, all_scores, threshold):  # 不算最佳threshold的话设为0.5
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    # p, r, _ = metrics.precision_recall_curve(all_trues, all_scores)
    # AUPR = metrics.auc(r, p)
    AUPR = metrics.average_precision_score(all_trues, all_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(all_trues, all_preds, labels=[0, 1]).ravel()
    return tp, tn, fp, fn,acc, f1, pre, rec, mcc, AUC, AUPR


def print_metrics(data_type, loss, metrics):
    """ Print the evaluation results """
    tp, tn, fp, fn, acc, f1, pre, rec, mcc, auc, aupr = metrics
    res = '\t'.join([
        '%s:' % data_type,
        'TP=%-5d' % tp,
        'TN=%-5d' % tn,
        'FP=%-5d' % fp,
        'FN=%-5d' % fn,
        'loss:%0.5f' % loss,
        'acc:%0.3f' % acc,
        'f1:%0.3f' % f1,
        'pre:%0.3f' % pre,
        'rec:%0.3f' % rec,
        'mcc:%0.3f' % mcc,
        'auc:%0.3f' % auc,
        'aupr:%0.3f' % aupr
    ])
    print(res)

def best_acc_thr(y_true, y_score):
    """ Calculate the best threshold with acc """
    best_thr = 0.5
    best_acc = 0
    for thr in range(1,100):
        thr /= 100
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(y_true, y_score, thr)
        if acc>best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def cal_by_epoch(config,mode, model, loader, loss, optimizer=None):
    # Model on train mode
    model.train() if mode == 'train' else model.eval()
    all_trues, all_scores = [], []
    losses, sample_num = 0.0, 0
    for iter_idx, (X, y) in enumerate(loader):
        sample_num += y.size(0)

        # Create vaiables
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.to(config.device).float())
            y_var = torch.autograd.Variable(y.to(config.device).float())

        # compute output
        model = model.to(config.device)
        y_pred = model(X_var).view(-1)

        # calculate and record loss
        loss_batch = loss(y_pred, y_var)
        losses += loss_batch.item()

        # compute gradient and do SGD step when training
        if mode == 'train':
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        # all_trues.append(y_var.data.cpu().numpy())
        # all_scores.append(output.data.cpu().numpy())
        all_trues.append(y.data.cpu().numpy())
        all_scores.append(y_pred.argmax(1).data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    return all_trues, all_scores, losses / sample_num

def train_one_epoch(config, model, train_loader, loss_func, optimizer, threshold):
    model.train()

    total_train_step = 0  # 记录训练的次数
    total_train_loss = 0
    start_time = time.time()

    all_trues = []
    all_scores = []
    sample_num=0
    # 训练模型
    for idx, data in enumerate(train_loader):
        X, y = data
        sample_num += y.size(0)
        # print('X的shape：', X.shape)   # torch.Size([64, 28, 28])   [batch_size, H, W]
        # print('y的shape:', y.shape)   # torch.Size([64, 1])         [batch_size, 1]
        X = X.unsqueeze(0)
        # print('X的shape：unsqueeze', X.shape)  # torch.Size([1, 64, 28, 28])  [channel,b,H,W]
        # y = y.long()
        # y = y.squeeze(1)              #[batch_size]
        # print('y.squeeze的shape：', y.shape)  # torch.Size([64])
        y = y.to(torch.float32)
        print("y.shape：", y.shape)  # torch.Size([64, 1])
        if torch.cuda.is_available():
            X = X.to(config.device)
            y = y.to(config.device)

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        total_train_loss += loss.item()  # 记录一轮的loss总值

        # 优化器
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        total_train_step += 1
        # =====================================================================
        # wandb.log({
        #     'train_one_epoch loss': loss
        # })
        # ======================================================================

        # 每一个epoch：隔24个iteration输出一次
        if total_train_step % 24 == 0:
            end_time = time.time()
            print("运行时间：", end_time - start_time)
            print("第{}次训练的loss={}  ".format(total_train_step, loss.item()))
        all_trues.append(y.data.cpu().numpy())
        # all_scores.append(y_pred.argmax(1).data.cpu().numpy())  # argmax(1),在轴1方向上找到最大值的下标（0或1）
        all_scores.append(y_pred.data.cpu().numpy())
    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores,threshold)
    tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores, threshold)
    return all_trues, all_scores, total_train_loss / sample_num, total_train_loss, acc, f1, pre, rec, mcc, AUC, AUPR


def val_one_epoch(config, model, val_loader, loss_func,threshold):
    model.eval()

    total_val_step = 0  # 记录测试的次数
    total_val_loss = 0
    all_trues = []
    all_scores = []
    sample_num = 0
    with torch.no_grad():
        for data in val_loader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有32个数据。
            X, y = data
            sample_num += y.size(0)
            X = X.unsqueeze(0)
            # y = y.long()
            # y = y.squeeze(1)  # torch.Size([32,])
            y = y.to(torch.float32)
            if torch.cuda.is_available():
                X = X.to(config.device)
                y = y.to(config.device)

            y_pred = model(X)
            loss = loss_func(y_pred, y)
            total_val_loss += loss.item()
            # print('测试集预测的概率：', y_pred)

            all_trues.append(y.data.cpu().numpy())
            # all_scores.append(y_pred.argmax(1).data.cpu().numpy())
            all_scores.append(y_pred.data.cpu().numpy())
        all_trues = np.concatenate(all_trues, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        # es = all_scores.sum()
        # print('test')
        matrix = confusion_matrix(all_trues, all_scores)
        print('confusion matrix:\n', matrix)
        # acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores,threshold)
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = compute_metrics(all_trues, all_scores,threshold)
    return all_trues, all_scores, total_val_loss / sample_num, total_val_loss, acc, f1, pre, rec, mcc, AUC, AUPR




def train_mode(config, model, train_loader, loss_func, optimizer, val_loader, epochs,threshold):
    # 配置wandb
    wandb_config = Namespace(
        project_name='sweep-test',
        batch_size=config.batch_size,
        lr=config.lr,
        epochs=config.epochs,
        dropout=config.dropout,
        weight_decay=config.weight_decay
    )

    # # 初始化wandb
    # wandb.init(
    #     project=wandb_config.project_name,
    #     config=wandb_config.__dict__,
    # )

    best_acc = 0.0
    save_path = ".\\"  # 当前目录下


    # 训练模型
    for epoch in range(epochs):
        print("------------------------第{}轮训练------------------------".format(epoch + 1))
        train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_one_epoch(
            config=config,
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            # wandb=wandb
            threshold=threshold
        )
        # ==========================================================
        # wandb.log({
        #     'train_epoch': epoch + 1,
        #     'train_loss': train_loss,
        #     'train_acc': train_acc,
        #     'train_f1': train_f1,
        #     'train_rec': train_rec,
        #     'train_mcc': train_mcc,
        #     'train_auc': train_auc,
        #     'train_aupr': train_aupr
        # })
        # =============================================================

        val_loss, val_acc, val_f1, val_pre, val_rec, val_mcc, val_auc, val_aupr = val_one_epoch(
            config=config,
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            threshold=threshold
        )

        # =============================================================
        # wandb.log({
        #     'val_epoch': epoch + 1,
        #     'val_loss': val_loss,
        #     'val_acc': val_acc,
        #     'val_f1': val_f1,
        #     'val_rec': val_rec,
        #     'val_mcc': val_mcc,
        #     'val_auc': val_auc,
        #     'val_aupr': val_aupr
        # })
        # ==============================================================

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
        torch.save(model.state_dict(), 'model_pytorch_test/model_{}.pth'.format(epoch + 1))

        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     print("save model{}".format(epoch + 1))
        #     torch.save(model.state_dict(), 'model_pytorch_test/model_{}.pth'.format(epoch + 1))
        #
        # print("第{}轮模型训练数据已保存".format(epoch + 1))

    # wandb.finish()




