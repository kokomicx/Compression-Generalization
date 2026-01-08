# main.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

from config import *
from utils import get_ring_matrix
from data_generator import SyntheticData
from solver import DCD_SGD_Trainer

def create_experiment_folder():
    """创建一个以当前时间命名的文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"experiments/{timestamp}_synthetic_stability"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Directory created: {folder_name}")
    return folder_name

def run_experiment(save_dir):
    print(f"Starting Experiment...")
    print(f"Topology: Ring, Agents: {NUM_AGENTS}, Dim: {DIM}")
    
    W = get_ring_matrix(NUM_AGENTS)
    
    # 存储所有结果
    # 结构: results[k] = {'stability': [], 'train_loss': [], 'test_loss': []}
    final_results = {}

    for k in K_LIST:
        ratio = k / DIM
        print(f"\nRunning for Top-{k} (Compression Ratio: {ratio:.2f})")
        
        # 临时列表存储多次 Trial 的结果
        trial_metrics = {
            'stability': [],
            'train_loss': [],
            'test_loss': []
        }
        
        for trial in tqdm(range(NUM_TRIALS), desc=f"Trials (k={k})"):
            gen = SyntheticData()
            data_S, data_S_prime = gen.generate_paired_datasets()
            test_data = gen.generate_test_data() # Generate test set
            
            trainer = DCD_SGD_Trainer(W, k)
            
            stab, train_l, test_l = trainer.run_experiment_step(data_S, data_S_prime, test_data)
            
            trial_metrics['stability'].append(stab)
            trial_metrics['train_loss'].append(train_l)
            trial_metrics['test_loss'].append(test_l)
        
        # 对 Trials 取平均
        final_results[k] = {
            'stability': np.mean(trial_metrics['stability'], axis=0),
            'train_loss': np.mean(trial_metrics['train_loss'], axis=0),
            'test_loss': np.mean(trial_metrics['test_loss'], axis=0)
        }
        
        # 保存该 k 下的 Raw Data (可选，方便后续分析)
        np.savez(f"{save_dir}/results_top_{k}.npz", 
                 stability=trial_metrics['stability'],
                 train_loss=trial_metrics['train_loss'],
                 test_loss=trial_metrics['test_loss'])

    return final_results

def plot_all_metrics(results, save_dir):
    """绘制三张图：Stability, Train Loss, Test Loss"""
    metrics_to_plot = [
        ('stability', r"Stability $\mathbb{E}[\|\theta_S - \theta_{S'}\|^2]$", 'stability.png'),
        ('train_loss', "Train Loss (MSE)", 'train_loss.png'),
        ('test_loss', "Test Loss (MSE)", 'test_loss.png')
    ]
    
    colors = ['r', 'g', 'b', 'k']
    styles = ['-', '-.', '--', ':']

    for metric_key, ylabel, filename in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for idx, k in enumerate(K_LIST):
            curve = results[k][metric_key]
            ratio = k / DIM
            label = f"Top-{k} (Ratio={ratio:.2f})"
            if k == DIM: label += " [No Comp]"
            
            plt.plot(curve, color=colors[idx % len(colors)], 
                     linestyle=styles[idx % len(styles)], linewidth=2, label=label)
        
        plt.title(f"{ylabel} vs. Iterations (Ring Topology)")
        plt.xlabel("Iterations (t)")
        plt.ylabel(ylabel)
        plt.yscale('log') # 统一用 Log Scale 观察收敛
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        plt.savefig(f"{save_dir}/{filename}", dpi=300)
        plt.close() # 关闭图像释放内存

    print(f"\nAll plots saved to {save_dir}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 1. 创建目录
    save_dir = create_experiment_folder()
    
    # 2. 运行实验
    results = run_experiment(save_dir)
    
    # 3. 画图
    plot_all_metrics(results, save_dir)