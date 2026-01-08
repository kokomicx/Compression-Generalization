# data_generator.py

import torch
from config import *

class SyntheticData:
    def __init__(self):
        # 1. 生成真实的权重 (Ground Truth)
        # 固定 Truth 以便计算 Excess Risk (可选)，这里主要用于生成数据
        self.true_w = torch.randn(DIM, device=DEVICE)
        self.true_w = self.true_w / torch.norm(self.true_w)

    def _generate_batch(self, num_samples):
        X = torch.randn(num_samples, DIM, device=DEVICE)
        noise = torch.randn(num_samples, device=DEVICE) * NOISE_LEVEL
        y = X @ self.true_w + noise
        return X, y

    def generate_paired_datasets(self):
        """
        生成训练数据 S 和 S'
        """
        data_S = []
        data_S_prime = []

        for agent_id in range(NUM_AGENTS):
            X, y = self._generate_batch(SAMPLES_PER_AGENT)
            data_S.append((X, y))

            if agent_id == 0:
                X_prime = X.clone()
                y_prime = y.clone()
                
                # 替换最后一个样本
                new_x, new_y = self._generate_batch(1)
                X_prime[-1] = new_x
                y_prime[-1] = new_y
                
                data_S_prime.append((X_prime, y_prime))
            else:
                data_S_prime.append((X, y))

        return data_S, data_S_prime

    def generate_test_data(self, test_samples=1000):
        """
        生成额外的测试数据 (Held-out validation set)
        为了方便计算，这里直接生成一个大 Batch，所有 Agent 共享评估
        或者模拟分布式测试数据。这里为了效率，返回一个大 Batch。
        """
        X_test, y_test = self._generate_batch(test_samples)
        return X_test, y_test