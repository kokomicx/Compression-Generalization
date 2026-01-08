# non-convex/main.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

from config import *
from data import get_cifar10_dataloaders
from agent import Agent
from models import ResNet20

def create_experiment_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(SAVE_DIR, f"{timestamp}_resnet_cifar")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def evaluate_consensus_model(agents, test_loader):
    """
    Evaluate the 'Virtual Consensus Model' (Average of all agents)
    This is strictly for plotting the Generalization Gap.
    """
    # 1. Create a temporary model to hold the average
    avg_model = ResNet20().to(DEVICE)
    avg_state_dict = avg_model.state_dict()
    
    # 2. Average parameters
    with torch.no_grad():
        for key in avg_state_dict.keys():
            avg_state_dict[key] = torch.stack([agent.model.state_dict()[key] for agent in agents]).mean(dim=0)
    avg_model.load_state_dict(avg_state_dict)
    
    # 3. Evaluate
    avg_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = avg_model(inputs)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / total, 100. * correct / total

def run_experiment(ratio, save_dir):
    print(f"\n=== Running Experiment: Top-{ratio*100}% Compression ===")
    
    # 1. Data & Agents
    train_loaders, test_loader = get_cifar10_dataloaders()
    agents = [Agent(i, train_loaders[i]) for i in range(NUM_AGENTS)]
    
    # 2. Topology (Ring)
    # Adjacency: i receives from i-1 and i+1
    # For CHOCO, we need to maintain neighbor estimates.
    # neighbor_estimates[i][j] = estimate of agent j held by agent i
    neighbor_estimates = [[
        [p.data.clone().detach() for p in agents[j].model.parameters()] 
        for j in range(NUM_AGENTS)] 
        for i in range(NUM_AGENTS)]
    
    # Metrics
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'bits': [],
        'epoch': []
    }
    
    total_bits_sent = 0
    global_step = 0
    
    # Iterations per epoch (assuming balanced data)
    steps_per_epoch = len(train_loaders[0])
    
    for epoch in range(EPOCHS):
        # Training Loop
        epoch_loss = 0
        
        with tqdm(total=steps_per_epoch, desc=f"Ep {epoch+1}/{EPOCHS}") as pbar:
            for _ in range(steps_per_epoch):
                global_step += 1
                
                # --- A. Local SGD Step ---
                for agent in agents:
                    loss = agent.train_one_step()
                    epoch_loss += loss
                
                # --- B. Compression & Communication ---
                # Each agent generates updates
                updates = [] # updates[i] contains the compressed diff from agent i
                step_bits = 0
                
                for agent in agents:
                    q, bits = agent.get_compressed_update(ratio)
                    updates.append(q)
                    step_bits += bits
                
                total_bits_sent += step_bits
                
                # --- C. Consensus Update (Ring Topology) ---
                # Agent i receives from (i-1)%N and (i+1)%N
                # CHOCO update: x_i = x_i + gamma * sum( (hat_x_j - hat_x_i) )
                
                with torch.no_grad():
                    # We first update the ESTIMATES held by neighbors
                    # Since we are simulating, 'neighbor_estimates[target][source]' needs to be updated
                    # with the 'updates[source]' we just calculated.
                    
                    # Update Estimates: Everyone receives everyone's broadcast (simplification for Ring implementation)
                    # or strictly only neighbors. 
                    # Strictly Ring: Agent i only sees updates from i-1 and i+1.
                    
                    for i in range(NUM_AGENTS):
                        left = (i - 1) % NUM_AGENTS
                        right = (i + 1) % NUM_AGENTS
                        neighbors = [left, right]
                        
                        # Apply updates to the estimates I hold of my neighbors
                        for neighbor_id in neighbors:
                            q_neighbor = updates[neighbor_id]
                            my_est_of_neighbor = neighbor_estimates[i][neighbor_id]
                            
                            for param_est, q in zip(my_est_of_neighbor, q_neighbor):
                                param_est.add_(q)
                        
                        # Also apply update to the estimate I hold of MYSELF (already done in get_compressed_update)
                        # But we need to make sure neighbor_estimates[i][i] is synced if we used it.
                        # Actually agent.hat_x_self is the authority.
                        
                        # --- Apply Consensus to Model Weights ---
                        # x_i += CONSENSUS_LR * ( sum_{j in N} (hat_x_j - hat_x_i) )
                        # Weight matrix W is usually 1/3 self, 1/3 left, 1/3 right.
                        # Standard CHOCO writes: x += gamma * Sum_j W_ij (hat_x_j - hat_x_i)
                        
                        agent_model_params = list(agents[i].model.parameters())
                        agent_self_est = agents[i].hat_x_self
                        
                        for p_idx, param in enumerate(agent_model_params):
                            # Self estimate
                            est_i = agent_self_est[p_idx]
                            
                            # Consensus force calculation
                            consensus_force = torch.zeros_like(param)
                            
                            # From Left
                            est_left = neighbor_estimates[i][left][p_idx]
                            consensus_force += (est_left - est_i)
                            
                            # From Right
                            est_right = neighbor_estimates[i][right][p_idx]
                            consensus_force += (est_right - est_i)
                            
                            # Averaging weight (1/3 for neighbors in strict Metropolice-Hastings, 
                            # but here we use a fixed CONSENSUS_LR as the mixing rate).
                            # Note: CONSENSUS_LR implies the 'gamma' in the paper.
                            # We normalize by degree typically or absorb into LR.
                            
                            param.add_(consensus_force, alpha=CONSENSUS_LR)

                pbar.update(1)

        # Learning Rate Scheduler Step
        for agent in agents:
            agent.scheduler.step()
            
        # --- D. Evaluation (End of Epoch) ---
        avg_train_loss = epoch_loss / (steps_per_epoch * NUM_AGENTS)
        
        # Evaluate Consensus Model on Test Set
        val_loss, val_acc = evaluate_consensus_model(agents, test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(val_loss)
        history['test_acc'].append(val_acc)
        history['bits'].append(total_bits_sent)
        history['epoch'].append(epoch)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Test Acc: {val_acc:.2f}% | Bits: {total_bits_sent/1e6:.1f} MB")
        
        # Save checkpoints periodically
        if epoch % 50 == 0:
            torch.save(agents[0].model.state_dict(), f"{save_dir}/model_top{ratio}_ep{epoch}.pth")
            
    # Save results
    np.savez(f"{save_dir}/results_top_{ratio}.npz", **history)
    return history

def plot_deep_results(all_results, save_dir):
    # Plot 1: Test Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    for ratio, res in all_results.items():
        plt.plot(res['epoch'], res['test_acc'], label=f"Top-{ratio*100}%", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"ResNet-20 on CIFAR-10 (Ring Topology)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/acc_vs_epoch.png")
    
    # Plot 2: Generalization Gap (Test Loss - Train Loss)
    plt.figure(figsize=(10, 6))
    for ratio, res in all_results.items():
        gap = np.array(res['test_loss']) - np.array(res['train_loss'])
        plt.plot(res['epoch'], gap, label=f"Top-{ratio*100}%", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Generalization Gap (Test - Train Loss)")
    plt.title("Implicit Regularization Evidence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/gen_gap.png")
    
    # Plot 3: The Trade-off (Test Acc vs Bits)
    plt.figure(figsize=(10, 6))
    for ratio, res in all_results.items():
        # Bits in Megabytes
        mb = np.array(res['bits']) / 8 / 1024 / 1024 
        plt.plot(mb, res['test_acc'], label=f"Top-{ratio*100}%", linewidth=2)
    plt.xlabel("Communication Volume (MB)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Communication Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/tradeoff.png")

if __name__ == "__main__":
    save_dir = create_experiment_folder()
    
    all_results = {}
    
    # Run for different compression ratios
    # Recommended: 0.01 (1%), 0.1 (10%), 1.0 (No Comp)
    for ratio in COMPRESSION_RATIOS:
        history = run_experiment(ratio, save_dir)
        all_results[ratio] = history
        
    plot_deep_results(all_results, save_dir)
    print(f"Done! Results saved to {save_dir}")