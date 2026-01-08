# utils.py

import torch
import networkx as nx
import numpy as np

def get_ring_matrix(n):
    """
    生成一个双随机 (Doubly Stochastic) 的 Ring 拓扑矩阵 W
    Self-loop weight = 1/3, Neighbor weight = 1/3
    """
    W = torch.zeros((n, n))
    for i in range(n):
        W[i, i] = 1/3
        W[i, (i - 1) % n] = 1/3
        W[i, (i + 1) % n] = 1/3
    return W

def top_k_compress(tensor, k):
    """
    Top-k 压缩算子.
    输入: tensor (shape: [dim])
    输出: compressed_tensor (shape: [dim]), mask
    """
    d = tensor.numel()
    if k >= d:
        return tensor.clone()
    
    # 获取绝对值最大的 top-k 索引
    _, indices = torch.topk(tensor.abs(), k)
    
    # 创建压缩后的稀疏向量
    compressed = torch.zeros_like(tensor)
    compressed[indices] = tensor[indices]
    
    return compressed