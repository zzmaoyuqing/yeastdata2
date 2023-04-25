'''
得到升序的蛋白质顺序
'''

import numpy as np
import pandas as pd

protein_emd = np.load('node2vec/output/protein_emd.npy')
protein_target = np.load('node2vec/output/protein_target.npy')
protein = np.load('node2vec/output/protein.npy')
# concat = np.concatenate([protein.reshape(-1, 1), protein_target, protein_emd], axis=1)

df = pd.DataFrame(protein_emd, index=protein.reshape(-1, 1))
# df.insert(0, 'protein', protein)
df.insert(0, 'target', protein_target)
df_sorted = df.sort_index()
# df_sorted = list(np.asarray(df_sorted))

np.save('data/sorted_protein_emd.npy', df_sorted.values[:, 1:])
# np.save('data/protein.npy', df_sorted[:, 0])     # 在划分数据集之前不需要单独输出target
