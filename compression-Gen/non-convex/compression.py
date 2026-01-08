# non-convex/compression.py

import torch

def layerwise_top_k_compress(param_tensor, ratio):
    """
    对单个 Tensor 进行 Top-k 压缩
    Returns: 
        compressed_tensor: 稀疏后的张量
        mask: 0/1 掩码
        bits: 传输所需的比特数 (Top-k values + Indices)
    """
    if ratio >= 1.0:
        # No compression
        bits = param_tensor.numel() * 32 # 32-bit floats
        return param_tensor, bits
    
    numel = param_tensor.numel()
    k = max(1, int(numel * ratio))
    
    # Top-k selection (absolute value)
    values, indices = torch.topk(param_tensor.abs().flatten(), k)
    
    # Create mask/sparse tensor
    # 实际传输的是：k个float32值 + k个int32索引
    # bits = k * (32 + 32) (简化计算，实际上索引可以用更少bit，但作为对比足够了)
    bits = k * 64 
    
    # 恢复稀疏 Tensor (用于模拟接收方)
    # 实际应用中会发 values 和 indices，这里直接用 mask 模拟
    threshold = values[-1] # 第 k 大的值
    mask = (param_tensor.abs() >= threshold).float()
    compressed_tensor = param_tensor * mask
    
    return compressed_tensor, bits