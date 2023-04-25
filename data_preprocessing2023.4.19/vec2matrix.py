import numpy as np
import pandas as pd

pro_emb = np.load('data/sorted_protein_emd_norm01.npy')

# 读取标准化后亚细胞数据，取前28列数据
sub = pd.read_excel('data/5093_sub_Standard_data.xlsx', 0, header=None)
sub_28 = sub.iloc[:, 1:29].values



def dot_emb_sub():
    a = []
    for i in range(len(pro_emb)):
        emb_sub = np.dot(pro_emb[i].reshape(28, 1), sub_28[i].reshape(1, 28))
        a.append(emb_sub)
    return a

np.save('data/dot_emb_sub', dot_emb_sub())

