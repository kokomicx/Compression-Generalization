# non-convex/config.py

import torch

# --- 硬件与随机性 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2024

# --- 分布式设置 ---
NUM_AGENTS = 5              # 深度学习比较耗显存，建议先用 5 或 10 个节点模拟
TOPOLOGY = 'ring'           # 'ring', 'torus', 'fully_connected'

# --- 数据集 CIFAR-10 ---
BATCH_SIZE = 32             # Local batch size
NUM_WORKERS = 2             # Dataloader workers

# --- 模型 ResNet-20 ---
# CIFAR-10 input is 32x32. ResNet-20 is standard for this.

# --- 优化器 (SGD) ---
LR = 0.1                    # Initial Learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# --- 训练控制 ---
EPOCHS = 200                # 标准 ResNet-20 训练通常跑 150-200 Epochs
LR_DECAY_EPOCHS = [100, 150]
LR_DECAY_RATE = 0.1

# --- DCD-SGD (CHOCO) 设置 ---
CONSENSUS_LR = 0.1          # 共识步长 (gamma/delta)，深度学习中通常取小一点
COMPRESSION_RATIOS = [0.01, 0.1, 1.0] # 1%, 10%, 100%

# --- 实验管理 ---
SAVE_DIR = "experiments"