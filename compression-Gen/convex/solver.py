# solver.py

import torch
from config import *
from utils import top_k_compress

class DCD_SGD_Trainer:
    def __init__(self, W, k):
        self.W = W.to(DEVICE) 
        self.k = k            
        
    def _init_states(self):
        x = torch.zeros(NUM_AGENTS, DIM, device=DEVICE)
        x_hat = torch.zeros(NUM_AGENTS, DIM, device=DEVICE)
        return x, x_hat

    def compute_gradient(self, x, data):
        grads = torch.zeros(NUM_AGENTS, DIM, device=DEVICE)
        for i in range(NUM_AGENTS):
            X, y = data[i]
            w_i = x[i]
            diff = X @ w_i - y
            # MSE Gradient + weak L2 regularization
            grad = (X.T @ diff) / SAMPLES_PER_AGENT + 0.001 * w_i
            grads[i] = grad
        return grads

    def evaluate_loss(self, x, X_data, y_data=None, is_distributed=True):
        """
        计算平均 Loss。
        如果 is_distributed=True, data 是 list of (X,y)
        如果 is_distributed=False, data 是 (X,y) 单一大矩阵 (Test set)
        """
        total_loss = 0.0
        
        if is_distributed:
            # Train Set Evaluation (Distributed)
            for i in range(NUM_AGENTS):
                X, y = X_data[i]
                w_i = x[i]
                # MSE Loss: 0.5 * ||Xw - y||^2 / m
                loss_i = 0.5 * torch.mean((X @ w_i - y)**2)
                total_loss += loss_i.item()
            return total_loss / NUM_AGENTS
        else:
            # Test Set Evaluation (Centralized/Shared)
            # 计算所有 Agent 在 Test Set 上的平均表现
            # Mean over Agents (Mean over Samples (Loss))
            # 也可以只取平均模型 \bar{x} 测，这里算所有个体模型的平均 Loss
            loss_sum = 0.0
            for i in range(NUM_AGENTS):
                w_i = x[i]
                loss_i = 0.5 * torch.mean((X_data @ w_i - y_data)**2)
                loss_sum += loss_i.item()
            return loss_sum / NUM_AGENTS

    def run_experiment_step(self, data_S, data_S_prime, test_data):
        """
        运行训练并返回: stability, train_loss, test_loss
        """
        x_S, x_hat_S = self._init_states()
        x_Sp, x_hat_Sp = self._init_states()
        
        X_test, y_test = test_data # Unpack test data
        
        # History logs
        stability_history = []
        train_loss_history = []
        test_loss_history = []

        for t in range(ITERATIONS):
            # --- System S ---
            g_S = self.compute_gradient(x_S, data_S)
            x_S = x_S - LEARNING_RATE * g_S
            
            q_S = torch.zeros_like(x_S)
            for i in range(NUM_AGENTS):
                q_S[i] = top_k_compress(x_S[i] - x_hat_S[i], self.k)
            x_hat_S = x_hat_S + q_S
            
            consensus_S = (self.W @ x_hat_S) - x_hat_S
            x_S = x_S + CONSENSUS_STEP_SIZE * consensus_S

            # --- System S' ---
            g_Sp = self.compute_gradient(x_Sp, data_S_prime)
            x_Sp = x_Sp - LEARNING_RATE * g_Sp
            
            q_Sp = torch.zeros_like(x_Sp)
            for i in range(NUM_AGENTS):
                q_Sp[i] = top_k_compress(x_Sp[i] - x_hat_Sp[i], self.k)
            x_hat_Sp = x_hat_Sp + q_Sp
            
            consensus_Sp = (self.W @ x_hat_Sp) - x_hat_Sp
            x_Sp = x_Sp + CONSENSUS_STEP_SIZE * consensus_Sp

            # --- Logging Metrics ---
            # 1. Stability
            sq_diff = torch.sum((x_S - x_Sp)**2, dim=1)
            avg_diff = torch.mean(sq_diff).item()
            stability_history.append(avg_diff)
            
            # 2. Train Loss (System S)
            train_loss = self.evaluate_loss(x_S, data_S, is_distributed=True)
            train_loss_history.append(train_loss)
            
            # 3. Test Loss (System S)
            test_loss = self.evaluate_loss(x_S, X_test, y_test, is_distributed=False)
            test_loss_history.append(test_loss)

        return stability_history, train_loss_history, test_loss_history