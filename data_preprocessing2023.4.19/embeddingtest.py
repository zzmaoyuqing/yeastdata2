import torch.nn as nn
import numpy as np
import torch

pro_emb = np.load('data/sorted_protein_emd.npy')
# Q0050 = torch.Tensor(pro_emb[0, :])



# 输入
# 规整后句子统一长度为2（每个句子2个词），3个句子，词和词id对应后，第一个句子为[1,4],第二个句子为[2,3]，第三个句子为[4,2]
# 如下1、2、4分别为三个句子中的第一个词，即序列中的第一个位置
# 4、3、2分别为三个句子中的第二个词，序列中的第二个位置
x = torch.LongTensor([[1, 2, 4], [4, 3, 2]])

# 调用nn.Embedding函数
# 一共5个词，每个词的词向量维度设置为6维
embeddings = nn.Embedding(5, 6, padding_idx=4)
print(embeddings(x))
print(embeddings(x).size())

# 输出
# [规整后的句子长度，样本个数（batch_size）,词向量维度]

