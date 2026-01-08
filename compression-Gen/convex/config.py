# config.py

import torch

# 实验环境设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
SEED = 42

# 拓扑与数据设置
NUM_AGENTS = 20          # N = 20
DIM = 500                # d = 500 (High dimension to make compression meaningful)
SAMPLES_PER_AGENT = 50   # 每个节点的样本数 (m)
NOISE_LEVEL = 0.1        # 标签噪声 (Least Squares y = wx + noise)

# 优化器设置
ITERATIONS = 500         # T
LEARNING_RATE = 0.01     # gamma
CONSENSUS_STEP_SIZE = 0.5 # delta (for CHOCO-SGD)

# 压缩设置
# Full dimension is 500. 
# k=10 (2%), k=50 (10%), k=100 (20%), k=500 (No Compression)
K_LIST = [ 50, 100, 500] 

# Stability 实验设置
NUM_TRIALS = 10          # 重复实验次数取平均，消除随机性