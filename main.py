import argparse
import os
import json
import time
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime

from models import ResNet20
from compressor import IdentityCompressor, TopKCompressor, QSGDCompressor
from node import Node
from utils import get_cifar10_loaders, get_ring_topology, plot_results

def evaluate_consensus(nodes, global_train_loader, test_loader, device):
    """
    Computes the Consensus Model \bar{x} and evaluates it.
    Returns: TrainLoss, ValLoss, TestAcc, ConsensusError
    """
    # 1. Compute Consensus Model parameters
    num_nodes = len(nodes)
    avg_params = torch.zeros_like(nodes[0].flat_params)
    
    # Calculate Consensus Error and Average simultaneously
    # Consensus Error = 1/N * sum(||x_i - x_bar||^2)
    # But we need x_bar first.
    
    with torch.no_grad():
        for node in nodes:
            avg_params += node.flat_params
        avg_params /= num_nodes
        
        # Compute Consensus Error
        consensus_error = 0.0
        for node in nodes:
            consensus_error += (node.flat_params - avg_params).norm().item() ** 2
        consensus_error /= num_nodes
    
    # 2. Load \bar{x} into a temporary model
    temp_model = ResNet20().to(device)
    
    # Helper to load flat params
    offset = 0
    with torch.no_grad():
        for p in temp_model.parameters():
            numel = p.numel()
            p.data.copy_(avg_params[offset:offset+numel].view_as(p))
            offset += numel
            
    temp_model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 3. Compute Training Loss (Empirical Risk)
    train_loss = 0.0
    train_total = 0
    with torch.no_grad():
        for inputs, targets in global_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = temp_model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)
    train_loss /= train_total
    
    # 4. Compute Validation Loss (Expected Risk) and Accuracy
    val_loss = 0.0
    correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = temp_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss /= val_total
    test_acc = 100. * correct / val_total
    
    return train_loss, val_loss, test_acc, consensus_error

def calculate_bits(compressor, num_params):
    if isinstance(compressor, IdentityCompressor):
        return 32 * num_params
    elif isinstance(compressor, TopKCompressor):
        k = max(1, int(num_params * compressor.ratio))
        # 32 bits for value, 32 bits for index
        return 32 * k + 32 * k
    elif isinstance(compressor, QSGDCompressor):
        # QSGD with bucketing
        # Bits = num_buckets * 32 (norms) + params * (1 sign + log2(levels))
        bucket_size = compressor.bucket_size
        num_buckets = math.ceil(num_params / bucket_size)
        bits_per_element = 1 + math.log2(compressor.num_levels)
        return num_buckets * 32 + num_params * bits_per_element
    return 32 * num_params

def run_experiment(exp_name, compressor, args, loaders, topology, device, output_dir, existing_results=None):
    print(f"\nRunning Experiment: {exp_name}")
    
    train_loaders, test_loader, global_train_loader = loaders
    num_nodes = args.num_nodes
    
    # Initialize Nodes
    nodes = []
    # Create one initial model to share initialization
    init_model = ResNet20().to(device)
    init_params = torch.cat([p.data.view(-1) for p in init_model.parameters()])
    
    for i in range(num_nodes):
        model = ResNet20().to(device)
        # Force same initialization
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(init_params[offset:offset+numel].view_as(p))
            offset += numel
            
        node = Node(i, model, train_loaders[i], device, compressor)
        nodes.append(node)
        
    # Initialize Topology
    for i in range(num_nodes):
        neighbors = []
        weights = {}
        for j in range(num_nodes):
            if topology[i, j] > 0:
                neighbors.append(j)
                weights[j] = topology[i, j].item()
        nodes[i].init_neighbors(neighbors, weights, init_params)
        
    # Experiment Loop
    results = [] if existing_results is None else existing_results
    cumulative_bits = 0.0
    bits_per_step = calculate_bits(compressor, init_params.numel())
    
    global_iter = 0
    for epoch in range(args.epochs):
        # LR Scheduler
        if epoch < 100:
            current_lr = args.lr
        elif epoch < 150:
            current_lr = args.lr * 0.1
        else:
            current_lr = args.lr * 0.01
            
        # Assume all loaders have same length (partitioned evenly)
        num_batches = len(train_loaders[0])
        
        for batch_idx in range(num_batches):
            
            # 1. Compute Gradients
            grads = {}
            for node in nodes:
                grads[node.node_id] = node.compute_gradient()

            # DEBUG: Check gradients for Node 0 at first iteration
            if global_iter == 0:
                 print(f"DEBUG: Node 0 Grad Norm: {grads[0].norm().item()}")
                
            # 2. Compute Update Step (Consensus -> Diff -> Compress -> Update)
            qs = {}
            for node in nodes:
                qs[node.node_id] = node.compute_update_step(grads[node.node_id], current_lr)
                
            # 3. Communicate (Update Neighbor Estimates)
            # In Ring, i sends to i-1 and i+1?
            # Topology matrix W_ij > 0 means i receives from j.
            # So if W_ij > 0, i needs q_j.
            # Broadcast model: everyone sends q to neighbors.
            for node in nodes:
                neighbor_qs = {}
                for nid in node.neighbors:
                    if nid != node.node_id:
                        neighbor_qs[nid] = qs[nid]
                node.update_neighbor_estimates(neighbor_qs)
                
            # Update metrics
            cumulative_bits += bits_per_step # Bits sent by ONE node (representative)
            global_iter += 1
            
            # Evaluation
            if global_iter % 50 == 0:
                print(f"Iter {global_iter}: Evaluating...", end='\r')
                train_loss, val_loss, test_acc, cons_err = evaluate_consensus(
                    nodes, global_train_loader, test_loader, device)
                
                gap = val_loss - train_loss
                
                log_entry = {
                    'Experiment': exp_name,
                    'Iteration': global_iter,
                    'Gap': gap,
                    'TrainLoss': train_loss,
                    'ValLoss': val_loss,
                    'TestAcc': test_acc,
                    'CommBits': cumulative_bits,
                    'ConsErr': cons_err
                }
                results.append(log_entry)
                
                # Live logging
                if global_iter % 200 == 0:
                    print(f"Exp: {exp_name} | Iter: {global_iter} | Loss: {train_loss:.4f} | Gap: {gap:.4f} | Acc: {test_acc:.2f}% | ConsErr: {cons_err:.4f}")
                    
    # Save intermediate results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    return results

def main():
    parser = argparse.ArgumentParser(description='DCD-SGD Generalization-Stability Trade-off')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per node')
    parser.add_argument('--num_nodes', type=int, default=5, help='Number of nodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_only', type=str, default=None, help='Run specific experiment (baseline, top10, top1, qsgd)')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = f"experiments/{timestamp}_DCD_SGD_Study"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save Config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # Data & Topology
    print("Preparing Data and Topology...")
    loaders = get_cifar10_loaders(args.num_nodes, args.batch_size)
    topology = get_ring_topology(args.num_nodes)
    
    results = []
    
    # 1. Baseline: Identity
    if args.run_only is None or args.run_only == 'baseline':
        print("\n=== Experiment 1/4: Baseline (Identity) ===")
        results = run_experiment("Baseline", IdentityCompressor(), args, loaders, topology, device, exp_dir, results)
    
    # 2. Top-10%
    if args.run_only is None or args.run_only == 'top10':
        print("\n=== Experiment 2/4: Top-10% ===")
        results = run_experiment("Top10", TopKCompressor(0.1), args, loaders, topology, device, exp_dir, results)
    
    # 3. Top-1%
    if args.run_only is None or args.run_only == 'top1':
        print("\n=== Experiment 3/4: Top-1% ===")
        results = run_experiment("Top1", TopKCompressor(0.01), args, loaders, topology, device, exp_dir, results)
    
    # 4. QSGD (levels=8)
    if args.run_only is None or args.run_only == 'qsgd':
        print("\n=== Experiment 4/4: QSGD (8 levels) ===")
        results = run_experiment("QSGD-8bit", QSGDCompressor(8), args, loaders, topology, device, exp_dir, results)
    
    # Final Plotting
    print("\nPlotting results...")
    plot_results(results, exp_dir, "All_Experiments")
    print(f"Done! Results saved to {exp_dir}")

if __name__ == '__main__':
    main()
