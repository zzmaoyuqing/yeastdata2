import numpy as np
from sklearn.preprocessing import MinMaxScaler

pro_data = np.load('data/sorted_protein_emd.npy')
pro_data_T = pro_data.T  # 因为MinMaxScaler是按列进行归一化处理的，所以先转置

Standard_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(pro_data_T)
Standard_data_T = Standard_data.T # 归一化之后再转置回去，才是pro_data归一化之后的数据
np.save('data/sorted_protein_emd_norm01.npy', Standard_data_T)  # 没有覆盖掉排好序的protein_emb.npy文件