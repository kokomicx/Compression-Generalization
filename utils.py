import torch
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import os
import pickle
import random

# --- Torchvision Fallback Implementation ---
# Since torchvision causes a segmentation fault in this environment,
# we implement the standard CIFAR-10 loading and transforms manually
# to match torchvision's behavior EXACTLY.

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    def __call__(self, pic):
        # pic is HWC numpy array, 0-255
        # Output: CHW tensor, 0-1
        # Match torchvision: pic.transpose((2, 0, 1)).float().div(255.0)
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255.0)
        return pic

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor):
        # tensor is CHW
        return (tensor - self.mean) / self.std

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        # img is numpy HWC
        if random.random() < self.p:
            return np.fliplr(img).copy()
        return img

class RandomCrop:
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding
        
    def __call__(self, img):
        # img is numpy HWC
        if self.padding is not None:
            # Pad with 0 (constant)
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            
        h, w, c = img.shape
        th, tw = self.size, self.size
        if w == tw and h == th:
            return img
            
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1+th, x1:x1+tw]

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        
        base_folder = os.path.join(root, 'cifar-10-batches-py')
        
        if self.train:
            file_list = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            file_list = ['test_batch']
            
        for file_name in file_list:
            file_path = os.path.join(base_folder, file_name)
            if not os.path.exists(file_path):
                 print(f"Warning: {file_path} not found. Ensure CIFAR-10 is downloaded.")
                 continue
            try:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                raise e
        
        if len(self.data) == 0:
            raise RuntimeError("No data loaded. Check ./data/cifar-10-batches-py path.")

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # img is numpy HWC, 0-255
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Ensure target is long tensor
        target = torch.tensor(target, dtype=torch.long)
            
        return img, target

class SyntheticDataset(Dataset):
    def __init__(self, size=200, mode='train', seed=42):
        super(SyntheticDataset, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Classes Y in {0, 1}
        self.targets = np.random.randint(0, 2, size=size)
        
        # Features X in R^2
        self.data = np.zeros((size, 2))
        
        mu0 = np.array([1, -1])
        mu1 = np.array([-1, 1])
        
        for i in range(size):
            if self.targets[i] == 0:
                self.data[i] = np.random.normal(mu0, 1.0)
            else:
                self.data[i] = np.random.normal(mu1, 1.0)
        
        # Label Noise (flip 10%)
        if mode == 'train':
            flip_mask = np.random.rand(size) < 0.1
            self.targets[flip_mask] = 1 - self.targets[flip_mask]
        
        # Convert to Tensor
        self.data = torch.FloatTensor(self.data)
        self.targets = torch.FloatTensor(self.targets) # Float for BCEWithLogitsLoss
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

def get_synthetic_loaders(num_nodes=20, batch_size=10):
    # Train: 20 nodes * 10 samples = 200 total
    train_dataset = SyntheticDataset(size=200, mode='train', seed=42)
    
    # Test: 10,000 samples
    test_dataset = SyntheticDataset(size=10000, mode='test', seed=2024)
    
    # Partition for Nodes
    train_loaders = []
    total_size = len(train_dataset)
    partition_size = total_size // num_nodes
    indices = list(range(total_size))
    
    for i in range(num_nodes):
        subset_indices = indices[i * partition_size : (i + 1) * partition_size]
        subset = Subset(train_dataset, subset_indices)
        train_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        
    # Global Train Loader (for evaluation)
    global_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    
    # Test Loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loaders, global_train_loader, test_loader

def get_topology_matrix(topo_type, n_nodes):
    W = torch.zeros((n_nodes, n_nodes))
    
    if topo_type == 'complete':
        W.fill_(1.0 / n_nodes)
        
    elif topo_type == 'identity':
        W = torch.eye(n_nodes)
        
    elif topo_type == 'ring':
        for i in range(n_nodes):
            W[i, i] = 1.0/3.0
            W[i, (i-1)%n_nodes] = 1.0/3.0
            W[i, (i+1)%n_nodes] = 1.0/3.0
            
    elif topo_type == 'lazy_complete':
        # W_ii = 0.95, W_ij = 0.05 / (n-1)
        off_diag_val = 0.05 / (n_nodes - 1)
        W.fill_(off_diag_val)
        for i in range(n_nodes):
            W[i, i] = 0.95
            
    else:
        raise ValueError(f"Unknown topology: {topo_type}")
        
    return W

def get_cifar10_loaders(num_nodes, batch_size, root='./data'):
    """
    Returns:
    - train_loaders: List of DataLoaders, one per node (partitioned training data)
    - test_loader: DataLoader for global test set
    - global_train_loader: DataLoader for global training set (for evaluation)
    """
    print("Preparing CIFAR-10 dataset (using manual loading due to torchvision segfault)...")
    
    # Standard CIFAR-10 Transforms
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load Datasets
    try:
        trainset = CIFAR10Dataset(root=root, train=True, transform=transform_train)
        testset = CIFAR10Dataset(root=root, train=False, transform=transform_test)
    except RuntimeError as e:
        print(f"Error loading CIFAR10: {e}")
        # Try to download if missing? 
        # Since we can't use torchvision.datasets.CIFAR10(download=True), 
        # we assume data is present or user needs to download it.
        # But wait, previous runs worked so data should be there.
        raise e

    # Global Loaders (Large batch size for fast evaluation)
    eval_batch_size = 256
    
    global_train_loader = DataLoader(
        trainset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        testset, batch_size=eval_batch_size, shuffle=False, num_workers=2)

    # Partitioning for Nodes
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices) # Optional: shuffle before partition
    
    split_size = num_train // num_nodes
    train_loaders = []
    
    for i in range(num_nodes):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_nodes - 1 else num_train
        node_indices = indices[start_idx:end_idx]
        
        subset = Subset(trainset, node_indices)
        
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2)
        train_loaders.append(loader)

    # Sanity Check for Data Loading
    print("Performing Data Loading Sanity Check...")
    try:
        sample_img, _ = next(iter(train_loaders[0]))
        print(f"Sanity Check - Image Shape: {sample_img.shape}")
        print(f"Sanity Check - Min Value: {sample_img.min().item():.4f}, Max Value: {sample_img.max().item():.4f}")
        if sample_img.max() > 50.0: # Should be around 2-3 after normalization
             print("WARNING: Max value is very high. Normalization might be wrong (e.g. not divided by 255).")
    except Exception as e:
        print(f"Sanity Check Failed: {e}")

    return train_loaders, test_loader, global_train_loader

def get_ring_topology(num_nodes):
    topology = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        topology[i, i] = 1.0
        topology[i, (i - 1) % num_nodes] = 1.0
        topology[i, (i + 1) % num_nodes] = 1.0
    topology /= 3.0
    return topology

def plot_results(results, output_dir, exp_name):
    """
    Plots results following publication standards:
    - Loss & Accuracy: Aggregated by Epoch (Mean) -> Smoother curves
    - Gap & Consensus: Raw per Iteration -> Stability analysis
    """
    if not results:
        return
    import matplotlib.pyplot as plt
    import pandas as pd
        
    df = pd.DataFrame(results)
    
    # Create Epoch-level aggregation for Loss and Accuracy
    # Ensure numeric only to avoid errors with non-numeric columns if any
    numeric_cols = ['TrainLoss', 'ValLoss', 'TestAcc']
    df_epoch = df.groupby('Epoch')[numeric_cols].mean()
    
    plt.figure(figsize=(12, 10))
    
    # 1. Loss (Aggregated by Epoch)
    plt.subplot(2, 2, 1)
    plt.plot(df_epoch.index, df_epoch['TrainLoss'], label='Train Loss')
    plt.plot(df_epoch.index, df_epoch['ValLoss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Per Epoch)')
    plt.legend()
    plt.grid(True)
    
    # 2. Accuracy (Aggregated by Epoch)
    plt.subplot(2, 2, 2)
    plt.plot(df_epoch.index, df_epoch['TestAcc'], label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy (Per Epoch)')
    plt.legend()
    plt.grid(True)
    
    # 3. Generalization Gap (Raw per Iteration)
    plt.subplot(2, 2, 3)
    plt.plot(df['Iteration'], df['Gap'], label='Generalization Gap', color='orange', alpha=0.6)
    plt.xlabel('Iteration')
    plt.ylabel('Gap (Val - Train)')
    plt.title('Generalization Gap (Raw)')
    plt.legend()
    plt.grid(True)
    
    # 4. Consensus Error (Raw per Iteration)
    plt.subplot(2, 2, 4)
    plt.plot(df['Iteration'], df['ConsErr'], label='Consensus Error', color='red', alpha=0.6)
    plt.xlabel('Iteration')
    plt.ylabel('Error Norm')
    plt.title('Consensus Error (Raw)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{exp_name}_plot.png'))
    plt.close()
