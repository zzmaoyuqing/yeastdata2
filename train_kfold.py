from utils2 import *
# from config import hyperparameter_defaults
# import wandb
import random
import vit
# from save import balanced_sampler
import MyData2
from torch.utils.data import DataLoader, SubsetRandomSampler
from pprint import pprint
from save import args

from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    # num_workers = 16
    # device = torch.device("cuda:" + str(args.gpu))
    # torch.cuda.set_device(args.gpu)
    num_workers = 0
    device = torch.device("cuda:" + str(0))
else:
    num_workers = 0
    device = torch.device("cpu")



def train(train_data, train_target, test_data, test_target):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # 设置随机种子
    setup_seed(42)

    # # 初始化wandb的config
    # wandb.init(config=hyperparameter_defaults, project='balanced_dataset_attention_sweep')
    # config = wandb.config
    config = args.args_parser()
    print(config)

    print(f'===================================New Training===================================')
    print("Device: ", config.device)
    print("Seed: ", config.seed)

    # 加载train_data
    dataset = MyData2.MyDataset(train_data, train_target, test_data, test_target)
    train_dataset, test_dataset = MyData2.TensorDataset(dataset)

    print("\n训练集的长度:{}".format(len(train_dataset)))
    print('number of essential protein in train_dataset:', int(train_dataset.tensors[1].sum()))
    print("测试集的长度:{}".format(len(test_dataset)))
    print('number of essential protein in test_dataset:', int(test_dataset.tensors[1].sum()))

    # loader data
    # test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(config.seed))



    # # 平衡采样train_data
    # print('{:-^40}'.format("train_data"))
    # balanced_data, balanced_target = balanced_sampler.balanced_dataset(train_data)
    # # 平衡采样val_data
    # print('{:-^40}'.format("val_data"))
    # val_balanced_data, val_balanced_target = balanced_sampler.balanced_dataset(val_data)
    #
    # balanced_sampler.list2tensor(balanced_data, balanced_target)
    # balanced_sampler.list2tensor(val_balanced_data, val_balanced_target)
    #
    # bd = balanced_sampler.MyBalancedDataset(balanced_data, balanced_target, val_balanced_data, val_balanced_target)
    #
    # balanced_train_dataset1, balanced_val_dataset1, balanced_train_dataset2, balanced_val_dataset2, \
    # balanced_train_dataset3, balanced_val_dataset3, balanced_train_dataset4, balanced_val_dataset4 = balanced_sampler.TensorDataset(
    #     bd)



    #
    # model = vit.ViT(
    #     image_size = 224,
    #     patch_size = 16,
    #     num_classes = 2,
    #     dim = 128,
    #     depth = 8,
    #     heads = 12,
    #     mlp_dim = 1024,
    #     dropout = 0.2,
    #     emb_dropout = 0.1
    # )


    model = vit.ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 1,
        dim = config.dim,
        depth = config.depth,
        heads = config.heads,
        mlp_dim = config.mlp_dim,
        dim_head = config.dim_head,
        dropout = config.dropout,
        emb_dropout = config.emb_dropout
    )


    if torch.cuda.is_available():
        model = model.to(config.device)

    loss_func, optimizer = compile(config, model)

    best_auc = 0.0
    patience = 25
    model_n=1 # 没有EP_EDL的集成策略，只有一个model
    save ='./model_pytorch'
    result_file = 'results.csv'

    # Train and validation using 5-fold cross validation
    val_auprs, test_auprs = [], []
    val_aucs, test_aucs = [], []
    test_trues, kfold_test_scores = [], []
    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=config.seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_dataset.tensors[0], train_dataset.tensors[1])):
        print(f'\nStart training CV fold {i + 1}:')
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False,
                                  num_workers=num_workers, worker_init_fn=np.random.seed(config.seed))
        val_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False,
                                num_workers=num_workers, worker_init_fn=np.random.seed(config.seed))


        # Train model
        count = 0
        best_val_aupr, best_test_aupr = .0, .0
        best_val_auc, best_test_auc = .0, .0
        best_test_scores = []
        best_model = model
        for epoch in range(config.epochs):
            print("------------------------第{}轮训练------------------------".format(epoch + 1))
            # Calculate prediction results and losses in [train_one_epoch()  val_one_epoch() val/test_one_epoch()]
            train_trues, train_scores, train_loss, total_train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_one_epoch(
                config=config,
                model=model,
                train_loader=train_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                # wandb=wandb
                threshold=config.threshold
            )
            # ==========================================================
            # wandb.log({
            #     'train_epoch': epoch + 1,
            #     'train_loss': train_loss,
            #     'train_acc': train_acc,
            #     'train_f1': train_f1,
            #     'train_pre': train_pre,
            #     'train_rec': train_rec,
            #     'train_mcc': train_mcc,
            #     'train_auc': train_auc,
            #     'train_aupr': train_aupr
            # })
            # =============================================================

            print('------------------------validataion -----------------------')
            val_trues, val_scores, val_loss, total_val_loss, val_acc, val_f1, val_pre, val_rec, val_mcc, val_auc, val_aupr = val_one_epoch(
                config=config,
                model=model,
                val_loader=val_loader,
                loss_func=loss_func,
                threshold=config.threshold
            )

            # =============================================================
            # wandb.log({
            #     'val_epoch': epoch + 1,
            #     'val_loss': val_loss,
            #     'val_acc': val_acc,
            #     'val_f1': val_f1,
            #     'val_pre': val_pre,
            #     'val_rec': val_rec,
            #     'val_mcc': val_mcc,
            #     'val_auc': val_auc,
            #     'val_aupr': val_aupr
            # })
            # ==============================================================
            print('------------------------test -----------------------')
            test_trues, test_scores, test_loss, total_test_loss, test_acc, test_f1, test_pre, test_rec, test_mcc, test_auc, test_aupr = val_one_epoch(
                config=config,
                model=model,
                val_loader=test_loader,
                loss_func=loss_func,
                threshold=config.threshold
            )
            # =============================================================
            # wandb.log({
            #     'test_epoch': epoch + 1,
            #     'test_loss': test_loss,
            #     'test_acc': test_acc,
            #     'test_f1': test_f1,
            #     'test_pre': test_pre,
            #     'test_rec': test_rec,
            #     'test_mcc': test_mcc,
            #     'test_auc': test_auc,
            #     'test_aupr': test_aupr
            # })
            # ==============================================================

            # Calculate evaluation meteics
            res = '\t'.join([
                '\nEpoch[%d/%d]' % (epoch + 1, config.epochs),
                '\nTrain dataset',
                'average loss:%0.5f' % train_loss,
                'total loss:%0.5f' % total_train_loss,
                'accuracy:%0.6f' % train_acc,
                'f-score:%0.6f' % train_f1,
                'precision:%0.6f' % train_pre,
                'recall:%0.6f' % train_rec,
                'mcc:%0.6f' % train_mcc,
                'auc:%0.6f' % train_auc,
                'aupr:%0.6f' % train_aupr,
                '\nVal dataset',
                'average loss:%0.5f' % val_loss,
                'total loss:%0.5f' % total_val_loss,
                'f-score:%0.6f' % val_f1,
                'precision:%0.6f' % val_pre,
                'recall:%0.6f' % val_rec,
                'mcc:%0.6f' % val_mcc,
                'auc:%0.6f' % val_auc,
                'aupr:%0.6f' % val_aupr,
                '\nTest dataset',
                'average loss:%0.5f' % test_loss,
                'total loss:%0.5f' % total_test_loss,
                'f-score:%0.6f' % test_f1,
                'precision:%0.6f' % test_pre,
                'recall:%0.6f' % test_rec,
                'mcc:%0.6f' % test_mcc,
                'auc:%0.6f' % test_auc,
                'aupr:%0.6f' % test_aupr,

            ])
            print(res)

            # # 早停止
            # early_stopping = EarlyStopping()
            # early_stopping(val_loss, model)
            # # 达到早停止条件时，early_stop会被置为True
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break  # 跳出迭代，结束训练

            # Sava the model by auc
            if val_auc > best_val_auc:
                count = 0
                best_model = model
                best_val_auc = val_auc
                best_val_aupr = val_aupr

                best_test_auc = test_auc
                best_test_aupr = test_aupr
                best_test_scores = test_scores

                print("!!!Get better model with valid AUC:{:.6f}. ".format(val_auc))

            else:
                count += 1
                if count >= patience:
                    torch.save(best_model, os.path.join(save,'model_{}_{:.3f}_{:.3f}.pkl'.format(i + 1, best_test_auc, best_test_aupr)))
                    print(f'Fold {i + 1} training done!!!\n')
                    break

        val_auprs.append(best_val_aupr)
        test_auprs.append(best_test_aupr)
        val_aucs.append(best_val_auc)
        test_aucs.append(best_test_auc)
        kfold_test_scores.append(best_test_scores)


    print(f'model training done!!!\n')
    for i, (test_auc, test_aupr) in enumerate(zip(test_aucs, test_auprs)):
        print('Fold {}: test AUC:{:.6f}   test AUPR:{:.6f}.'.format(i+1, test_auc, test_aupr))

    # Average 5 models' results
    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold

    # Cal the best threshold
    best_acc_threshold, best_acc = best_acc_thr(test_trues, final_test_scores)
    print('The best acc threshold is {:.2f} with the best acc({:.3f}).'.format(best_acc_threshold, best_acc))

    # Select the best threshold by acc
    final_test_metrics = compute_metrics(test_trues, final_test_scores, best_acc_threshold)[:]
    print_metrics('Final test', test_loss, final_test_metrics)



            # wandb.finish()


if __name__ == '__main__':
    train_data = np.load('dataset/dim224/train_test/train_set/train_data.npy')

    train_target = np.load('dataset/dim224/train_test/train_set/train_target.npy')

    test_data = np.load('dataset/dim224/train_test/test_set/test_data.npy')
    test_target = np.load('dataset/dim224/train_test/test_set/test_target.npy')
    train(train_data, train_target, test_data, test_target)
    # # 配置sweep超参数搜索方法和指标
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {
    #         'goal': 'maximize',
    #         'name': 'val_auc'
    #     }
    # }
    #
    # # 配置sweep超参数范围
    # parameters_dict = {
    #     # 离散型超参
    #     'epochs': {
    #         'values': [50, 80, 110]
    #     },
    #     'head':{
    #         'values': [1, 2, 3, 4]
    #     },
    #     # 连续性分布超参
    #     'batch_size': {
    #         'distribution': 'q_uniform',
    #         'q': 8,  # 8维间隔作均匀分布
    #         'min': 64,
    #         'max': 256
    #     },
    #     'dropout': {
    #         'distribution': 'uniform',
    #         'min': 0,
    #         'max': 0.6
    #     },
    #     'lr': {
    #         'distribution': 'log_uniform_values',
    #         'min': 1e-6,
    #         'max': 0.1
    #     },
    #
    #     'weight_decay': {
    #         'distribution': 'uniform',
    #         'min': 1e-6,
    #         'max': 0.1
    #     }
    # }
    # sweep_config['parameters'] = parameters_dict
    # pprint(sweep_config)
    # # 初始化sweep controller
    # sweep_id = wandb.sweep(sweep_config, project='balanced_dataset_attention_sweep')
    #
    # # agent
    # wandb.agent(sweep_id, train, count=100)