# -*- coding:utf-8 -*-

import numpy as np
from args import args_parser
from node2vec import node2vec
import networkx as nx
import pandas as pd
import torch as t


def main():

    args = args_parser()
    # 保证节点和标签一一对应
    PPI_graph = nx.read_edgelist('data/yeast_PPI5093.txt')
    G = nx.Graph()
    G.add_nodes_from(sorted(PPI_graph.nodes(data=True)))
    G.add_edges_from(PPI_graph.edges(data=True))

    # 向图中的edge加入weight属性
    for u, v in G.edges:  # 为PPI网络加权重
        G.add_edge(u, v, weight=1)

    # 向图中的node加入target属性
    import pandas as pd
    data = pd.read_csv("data/protein_target.txt")
    target = data.iloc[:, 2]
    target_mapping = dict(zip(G.nodes, target))
    for node in G.nodes:
        G.nodes[node]['y_target'] = target_mapping[node]

   # print("G.nodes ", G.nodes)
   # print("G.edges ", G.edges)

    vec = node2vec(args, G)

    nodes, walks = vec.get_walks()                              # 获得nodes和walks
    #print("walks的shape", np.array(walks).shape)

    embeddings = vec.learning_features(nodes, walks)            # 获得每个node的embedding
    targets = [list(G.nodes[node].values()) for node in nodes]  # 获得每个embedding对应其node的标签
    print("nodes对应的targets：", targets)
    #print(embeddings)
    print('embeddings的shape', np.array(embeddings).shape)
    np.save('output/protein_emd.npy', embeddings)        # 测试时先备份这三个npy，否则会覆盖之前的原数据
    np.save('output/protein_target.npy', targets)
    np.save('output/protein.npy', nodes)


if __name__ == '__main__':
    main()