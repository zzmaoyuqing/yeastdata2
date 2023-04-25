# -*- coding: utf-8 -*-

import numpy as np
import sys
import random

import pandas as pd

sys.path.append('../')
import numpy.random as npr
from gensim.models import Word2Vec

'''总结一下node2vec的模型框架：通过二阶random walk计算每个节点到其他节点的概率，然后将这个概率输出为两组数，方便后面抽样调用。'''




class node2vec:
    def __init__(self, args, G):
        self.G = G
        self.args = args
        self.init_transition_prob()

    def init_transition_prob(self):
        """
        :return:Normalized transition probability matrix
        """
        g = self.G
        nodes_info, edges_info = {}, {}


        for node in g.nodes:
            nbs = sorted(g.neighbors(node))
            probs = [g[node][n]['weight'] for n in nbs]
            # Normalized
            norm = sum(probs)
            normalized_probs = [float(n) / norm for n in probs]
            nodes_info[node] = self.alias_setup(normalized_probs)

        for edge in g.edges:
            # directed graph
            if g.is_directed():
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1])
            # undirected graph
            else:
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1])
                edges_info[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.nodes_info = nodes_info
        self.edges_info = edges_info

    #  论文中figure2的代码实现：得到节点到节点的概率(对于不是第二次采样的情况)
    def get_alias_edge(self, t, v):
        """
        Get the alias edge setup lists for a given edge.
        """
        g = self.G
        unnormalized_probs = []
        for v_nbr in sorted(g.neighbors(v)):
            if v_nbr == t:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / self.args.p)
            elif g.has_edge(v_nbr, t):
                unnormalized_probs.append(g[v][v_nbr]['weight'])
            else:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / self.args.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    # 采样策略：alias_setup的作用是根据二阶random walk输出的概率变成每个节点对应两个数，被后面的alias_draw函数所进行抽样
    def alias_setup(self, probs):
        """
        :probs: v到所有x的概率
        :return: Alias and Prob
        """
        K = len(probs)
        q = np.zeros(K)                  # 对应Prob数组
        J = np.zeros(K, dtype=np.int)    # 对应Alias数组
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []  # 存储比1小的列
        larger = []   # 存储比1大的列
        for kk, prob in enumerate(probs):
            q[kk] = K * prob  #
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.

        #
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large  #
            q[large] = q[large] - (1.0 - q[small])  #

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    # 使用别名采样从非均匀离散分布中抽取样本
    def alias_draw(self, J, q):
        """
        in: Prob and Alias
        out: sampling results
        """
        K = len(J)
        # Draw from the overall uniform mixture.
        kk = int(np.floor(npr.rand() * K))  # random

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if npr.rand() < q[kk]:  # compare
            return kk
        else:
            return J[kk]

    # 对于给定的长度，对于开起始节点开始模拟这个节点的路径
    def node2vecWalk(self, u):
        g = self.G
        walk = [u]
        nodes_info, edges_info = self.nodes_info, self.edges_info
        while len(walk) < self.args.l:
            curr = walk[-1]
            v_curr = sorted(g.neighbors(curr))
            if len(v_curr) > 0:
                if len(walk) == 1:
                    # print(adj_info_nodes[curr])
                    # print(alias_draw(adj_info_nodes[curr][0], adj_info_nodes[curr][1]))
                    walk.append(v_curr[self.alias_draw(nodes_info[curr][0], nodes_info[curr][1])])          # alias_draw这个函数是等于是根据二阶random walk概率选择下一个点
                else:
                    prev = walk[-2]
                    ne = v_curr[self.alias_draw(edges_info[(prev, curr)][0], edges_info[(prev, curr)][1])]
                    walk.append(ne)
            else:
                break

        return walk

    def get_walks(self):
        walks = []
        g = self.G
        nodes = list(g.nodes())
        # print("未shuffle的图节点：",nodes)
        for t in range(self.args.r):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vecWalk(node)           # 每个节点生成walk sequence
                walks.append(walk)
        # embedding
        print('get_walks得到图的节点：', nodes)             # 每次生成图的节点是不一样的
        walks = [list(map(str, walk)) for walk in walks]  # walks中存储中每一个节点的r条长为l的随机游走路径。

        print("walks的shape", np.array(walks).shape)
        print('第一个节点的第一条随机游走序列：', walks[0])
        print('第一个节点的第二条随机游走序列：', walks[1])
        print('第二个节点的第一条随机游走序列：', walks[11])
        return nodes, walks


    def learning_features(self,nodes,walks):
        # 有了walks之后，利用gensim库中的Word2Vec进行训练，进而得到所有节点的向量表示。
        # Word2Vec的参数：walks是采样的结果；vector_size是节点向量的维度，这里为d;sg=1 设置为skip-gram model；训练并行数，这里选择3
        model = Word2Vec(sentences=walks, vector_size=self.args.d, window=self.args.k, min_count=0, sg=1, workers=3)
        f = model.wv   # f中存储着所有节点的长度为d的向量表示
        res = [f[x] for x in nodes]
        return res




