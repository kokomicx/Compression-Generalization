import torch
import torch.nn.functional as F

class Node:
    def __init__(self, node_id, model, loader, device, compressor):
        self.node_id = node_id
        self.model = model
        self.loader = loader
        self.device = device
        self.compressor = compressor
        self.iterator = iter(self.loader)
        
        # Initialize flat parameters
        self.flat_params = self._flatten_params()
        
        # Momentum state
        self.velocities = torch.zeros_like(self.flat_params)
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Neighbor tracking
        self.neighbor_estimates = {} # {neighbor_id: flat_tensor}
        self.neighbors = [] # List of neighbor IDs
        self.weights = {} # {neighbor_id: weight}

    def _flatten_params(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _load_params(self, flat_params):
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def init_neighbors(self, neighbors, weights, init_params):
        """
        neighbors: list of neighbor IDs (including self if in topology)
        weights: dict {neighbor_id: weight}
        init_params: flat tensor of initial parameters
        """
        self.neighbors = neighbors
        self.weights = weights
        
        # Initialize neighbor estimates
        # We also need a local estimate for self (x^t_i)
        # In DCD-SGD, x^t_i is the "public" state that neighbors have
        self.local_estimate = init_params.clone().to(self.device)
        
        for nid in neighbors:
            if nid != self.node_id:
                self.neighbor_estimates[nid] = init_params.clone().to(self.device)
        
        # Ensure our params are on device
        self.flat_params = self.flat_params.to(self.device)
        self.velocities = self.velocities.to(self.device)
        self._load_params(self.flat_params)

    def compute_gradient(self):
        self.model.train()
        try:
            inputs, targets = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            inputs, targets = next(self.iterator)
            
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Manually zero gradients
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                
        outputs = self.model(inputs)
        
        # [FIX] Automatically detect task type
        if outputs.shape[1] == 1:
            # Binary Classification (Synthetic Logistic Regression)
            # Ensure target is float and matches output dim (N, 1)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs, targets.float().view_as(outputs))
        else:
            # Multi-class (CIFAR-10)
            loss = F.cross_entropy(outputs, targets)
            
        loss.backward()
        
        # Clip gradients to prevent explosion (Relaxed to 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # Flatten gradient
        grads = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p.view(-1)) for p in self.model.parameters()])
        return grads

    def compute_update_step(self, grads, lr):
        """
        Performs the DCD-SGD update step:
        1. Consensus: x_half = sum(W_ij * x_j_hat) - lr * g
        2. Diff: z = x_half - x_t
        3. Compress: q = C(z)
        4. Update: x_{t+1} = x_t + q
        """
        with torch.no_grad():
            # 1. Consensus
            consensus = torch.zeros_like(self.flat_params)
            for nid in self.neighbors:
                w = self.weights[nid]
                if nid == self.node_id:
                    est = self.local_estimate
                else:
                    est = self.neighbor_estimates[nid]
                consensus += w * est
            
            # Apply Weight Decay
            grads = grads.add(self.flat_params, alpha=self.weight_decay)
            
            # Update Velocity
            self.velocities = self.momentum * self.velocities + grads
            
            # Compute Update Step using Velocity
            update_step = lr * self.velocities
            
            x_half = consensus - update_step
            
            # 2. Diff
            z = x_half - self.flat_params
            
            # 3. Compress
            q = self.compressor.compress(z)
            
            # 4. Update
            self.flat_params += q
            self.local_estimate += q
            self._load_params(self.flat_params)
            
            return q

    def update_neighbor_estimates(self, neighbor_qs):
        """
        Update estimates of neighbors based on received compressed differences.
        neighbor_qs: dict {neighbor_id: q_tensor}
        """
        with torch.no_grad():
            for nid, q in neighbor_qs.items():
                if nid in self.neighbor_estimates:
                    self.neighbor_estimates[nid] += q
