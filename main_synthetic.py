import argparse
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from models import LogisticRegression
from compressor import IdentityCompressor
from node import Node
from utils import get_synthetic_loaders, get_topology_matrix

def evaluate_synthetic(nodes, global_train_loader, test_loader, device):
    """
    Evaluates Consensus Model on Synthetic Data
    """
    # 1. Compute Consensus Model
    num_nodes = len(nodes)
    avg_params = torch.zeros_like(nodes[0].flat_params)
    
    with torch.no_grad():
        for node in nodes:
            avg_params += node.flat_params
        avg_params /= num_nodes
        
        # Consensus Error
        consensus_error = 0.0
        for node in nodes:
            consensus_error += (node.flat_params - avg_params).norm().item() ** 2
        consensus_error /= num_nodes
        
    # 2. Load into Temp Model
    temp_model = LogisticRegression(input_dim=2, output_dim=1).to(device)
    
    offset = 0
    with torch.no_grad():
        for p in temp_model.parameters():
            numel = p.numel()
            p.data.copy_(avg_params[offset:offset+numel].view_as(p))
            offset += numel
            
    temp_model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Train Loss (Risk on Sample)
    train_loss = 0.0
    train_total = 0
    with torch.no_grad():
        for inputs, targets in global_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = temp_model(inputs).squeeze()
            loss = criterion(outputs, targets)
            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)
    train_loss /= train_total
    
    # 4. Test Loss (Risk on Population)
    test_loss = 0.0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = temp_model(inputs).squeeze()
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            test_total += targets.size(0)
    test_loss /= test_total
    
    return train_loss, test_loss, consensus_error

def run_synthetic_experiment(topo_type, num_nodes, iterations, lr, batch_size, output_dir):
    print(f"\nRunning Synthetic Experiment: Topology={topo_type}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Data & Topology
    train_loaders, global_train_loader, test_loader = get_synthetic_loaders(num_nodes, batch_size=batch_size)
    topology = get_topology_matrix(topo_type, num_nodes)
    
    # 2. Initialize Nodes
    nodes = []
    # Zero initialization
    init_model = LogisticRegression(input_dim=2, output_dim=1).to(device)
    for p in init_model.parameters():
        p.data.fill_(0.0)
        
    # Neighbors map
    neighbor_maps = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if topology[i, j] > 0:
                neighbor_maps[i].append(j)

    for i in range(num_nodes):
        node = Node(
            node_id=i,
            model=LogisticRegression(input_dim=2, output_dim=1).to(device),
            loader=train_loaders[i],
            device=device,
            compressor=IdentityCompressor()
        )
        
        # Get neighbors and weights for this node
        my_neighbors = []
        my_weights = {}
        for j in range(num_nodes):
            if topology[i, j] > 0:
                my_neighbors.append(j)
                my_weights[j] = topology[i, j].item()
        
        # Initialize neighbors
        # Need a flat tensor for init_params
        flat_init = torch.cat([p.data.view(-1) for p in init_model.parameters()])
        node.init_neighbors(my_neighbors, my_weights, flat_init)
        
        # Override node momentum to 0 for vanilla SGD in synthetic exp
        # And weight decay to 0
        node.momentum = 0.0
        node.weight_decay = 0.0
        
        nodes.append(node)
        
    results = []
    
    # 3. Training Loop
    for t in range(iterations):
        # Compute Gradients
        grads = {}
        for node in nodes:
            grads[node.node_id] = node.compute_gradient()
            
        # Update Step (DCD)
        qs = {}
        for node in nodes:
            # Note: We use constant LR = 0.03
            qs[node.node_id] = node.compute_update_step(grads[node.node_id], lr)
            
        # Communication
        for node in nodes:
            neighbor_qs = {}
            for nid in node.neighbors:
                if nid != node.node_id:
                    neighbor_qs[nid] = qs[nid]
            node.update_neighbor_estimates(neighbor_qs)
            
        # Evaluation
        if t % 10 == 0 or t == iterations - 1:
            train_loss, test_loss, cons_err = evaluate_synthetic(
                nodes, global_train_loader, test_loader, device)
            
            gap = test_loss - train_loss
            
            print(f"Iter {t} | TrainL: {train_loss:.4f} | TestL: {test_loss:.4f} | Gap: {gap:.4f} | ConsErr: {cons_err:.4f}")
            
            results.append({
                'Topology': topo_type,
                'Iteration': t,
                'TrainLoss': train_loss,
                'TestLoss': test_loss,
                'Gap': gap,
                'ConsErr': cons_err
            })
            
    return results

def plot_synthetic_results(all_results, output_dir):
    df = pd.DataFrame(all_results)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Train Loss
    plt.subplot(1, 3, 1)
    for topo in df['Topology'].unique():
        sub = df[df['Topology'] == topo]
        plt.plot(sub['Iteration'], sub['TrainLoss'], label=topo)
    plt.title('Train Loss (Optimization)')
    plt.xlabel('Iteration')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Generalization Gap
    plt.subplot(1, 3, 2)
    for topo in df['Topology'].unique():
        sub = df[df['Topology'] == topo]
        plt.plot(sub['Iteration'], sub['Gap'], label=topo)
    plt.title('Generalization Gap (Test - Train)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # 3. Consensus Error
    plt.subplot(1, 3, 3)
    for topo in df['Topology'].unique():
        sub = df[df['Topology'] == topo]
        plt.plot(sub['Iteration'], sub['ConsErr'], label=topo)
    plt.title('Consensus Error')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_results.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of agents')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=500, help='Total training iterations')
    parser.add_argument('--topology', type=str, default='all', help="Topology: 'complete', 'identity', 'ring', 'lazy_complete', or 'all'")
    parser.add_argument('--batch_size', type=int, default=10, help='Local batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f"experiments/{timestamp}_Nodes{args.num_nodes}_LR{args.lr}_{args.topology}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.topology == 'all':
        topologies = ['complete', 'identity', 'ring', 'lazy_complete']
    else:
        topologies = [args.topology]
    
    all_results = []
    
    for topo in topologies:
        res = run_synthetic_experiment(
            topo, 
            num_nodes=args.num_nodes, 
            iterations=args.iterations, 
            lr=args.lr, 
            batch_size=args.batch_size,
            output_dir=output_dir
        )
        all_results.extend(res)
        
    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Plot
    plot_synthetic_results(all_results, output_dir)
    print(f"Done! Results saved to {output_dir}")

if __name__ == '__main__':
    main()
