# non-convex/agent.py

import torch
import torch.optim as optim
from models import ResNet20
from compression import layerwise_top_k_compress
from config import *

class Agent:
    def __init__(self, id, train_loader):
        self.id = id
        self.train_loader = train_loader
        self.data_iterator = iter(train_loader)
        
        # 1. Local Model & Optimizer
        self.model = ResNet20().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=LR_DECAY_EPOCHS, gamma=LR_DECAY_RATE)
        
        # 2. CHOCO State: Estimates of neighbors (Simulation: we store locally)
        # self.hat_x[j] stores my estimate of neighbor j's model
        # For simplicity in simulation: we will access neighbors' buffers directly in main loop
        # But we need our own 'hat_x_self' (My estimate of myself broadcasted to others)
        self.hat_x_self = [p.data.clone().detach() for p in self.model.parameters()]
        
        # Buffer for error accumulation (Implicitly handled by model - hat_x_self)
        
    def train_one_step(self):
        """Standard SGD Step"""
        try:
            inputs, targets = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_loader)
            inputs, targets = next(self.data_iterator)
            
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_compressed_update(self, ratio):
        """
        CHOCO Step: Compress(x - hat_x)
        Returns: 
            updates: List of compressed tensors (one per layer)
            total_bits: Communication cost
        """
        updates = []
        total_bits = 0
        
        with torch.no_grad():
            for param, hat_param in zip(self.model.parameters(), self.hat_x_self):
                # Calculate difference
                diff = param.data - hat_param.data
                
                # Compress
                compressed_diff, bits = layerwise_top_k_compress(diff, ratio)
                
                # Update my own estimate (simulate sending)
                hat_param.data += compressed_diff
                
                updates.append(compressed_diff)
                total_bits += bits
                
        return updates, total_bits

    def update_consensus(self, neighbor_updates, consensus_lr):
        """
        Consensus Step: x = x + gamma * Sum(hat_x_neighbor - hat_x_self)
        But since we received 'compressed_update' (q_j), we know:
        New_Hat_Neighbor = Old_Hat_Neighbor + q_j
        
        Mathematical shortcut for CHOCO implementation:
        x_new = x_old + gamma * Sum( (Old_Hat_j + q_j) - (Old_Hat_i + q_i) )
        ... actually, simpler implementation in code:
        
        We maintain a `consensus_buffer` which accumulates the weighted estimates.
        """
        # In this simulation, we will apply updates directly to the model parameters
        # x <- x + gamma * (Neighbor_Hat - My_Hat)
        pass # Implemented in main loop for easier interaction handling