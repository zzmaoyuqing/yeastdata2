1.PPI（5093）：由node2vec得到PPI的embedding   维度为28    ---结果在output文件夹下（每一次生成的embedding不一样，所以我以2023.4.8/node2vec/output里的三个npy文件为主要数据）
2.将PPI生成的embedding排序              sort_protein.py                 （protein_emd.npy(原)、protein_target.npy、protein.npy-->sorted_protein_emd.npy）
3.将排序后的embedding归一化到（0，1）   normalized_pro_emb.py                 （'data/sorted_protein_emd.npy‘ ---->sorted_protein_emd_norm01.npy）
4.emb和sub（取前28列）做内积            vec2matrix.py   （5093_sub_Standard_data.xlsx、sorted_protein_emb_norm01.npy、protein_target.txt-->dot_emb_sub.npy)
5.封装成Dataset类：
划分数据集0.6：.2：0.2                         MyData.py                 （dot_emb_sub.npy、protein_target.txt-->train_set、test_set、val_set下所有文件）



亚细胞（从数据库下载）：之前处理过了 直接拿5093_sub_Standard_data.xlsx来用
1.将subcellular归一化处理     normalized_subcellular.py  亚细胞归一化处理  （yeast_compartment_integrated_full.tsv--> sub_Standard_data.csv）
2.取PPI网络中的subcellular         matlab/subdata1_1.m   过滤上面输出的文件（sub_Standard_data.csv、yeast_PPI5093.txt、sub_location.xlsx-->5093_sub_Standard_data.xlsx）

