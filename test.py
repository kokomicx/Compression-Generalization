from utils import get_synthetic_loaders
import torch
import numpy as np

train_loaders, _, _ = get_synthetic_loaders(num_nodes=20, batch_size=10)

# 检查 Node 0 和 Node 1 的第一个 Batch 是否一样
iter0 = iter(train_loaders[0])
iter1 = iter(train_loaders[1])

data0, target0 = next(iter0)
data1, target1 = next(iter1)

print("Data 0 Mean:", data0.mean().item())
print("Data 1 Mean:", data1.mean().item())

if torch.allclose(data0, data1):
    print("CRITICAL ALARM: Node 0 and Node 1 have IDENTICAL data!")
else:
    print("Data is different. Good.")
