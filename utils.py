import torch
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

# --- Custom Transforms & Dataset to avoid torchvision segfault ---

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
        img = torch.from_numpy(pic.transpose((2, 0, 1))).float().div(255.0)
        return img

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
        if np.random.random() < self.p:
            return np.fliplr(img).copy()
        return img

class RandomCrop:
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding
        
    def __call__(self, img):
        # img is numpy HWC
        if self.padding is not None:
            # Pad H and W, not C
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            
        h, w, c = img.shape
        th, tw = self.size, self.size
        if w == tw and h == th:
            return img
            
        x1 = np.random.randint(0, w - tw + 1)
        y1 = np.random.randint(0, h - th + 1)
        return img[y1:y1+th, x1:x1+tw]

class CustomCIFAR10(Dataset):
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
                print("Please ensure CIFAR-10 is downloaded in './data'.")
                raise e
                    
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target
        
    def __len__(self):
        return len(self.data)

# --- End Custom Utils ---

def get_cifar10_loaders(num_nodes, batch_size, root='./data'):
    print("Initializing Custom Transforms...")
    # Transforms
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

    print("Loading Custom Trainset...")
    # Datasets
    # Assumes data is already present in root/cifar-10-batches-py
    trainset = CustomCIFAR10(root=root, train=True, transform=transform_train)
    
    print("Loading Custom Testset...")
    testset = CustomCIFAR10(root=root, train=False, transform=transform_test)
    
    print("Creating Global Loader...")
    global_train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Creating Test Loader...")
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Partitioning
    print("Partitioning...")
    total_size = len(trainset)
    indices = list(range(total_size))
    split_size = total_size // num_nodes
    
    train_loaders = []
    for i in range(num_nodes):
        start = i * split_size
        end = (i + 1) * split_size if i < num_nodes - 1 else total_size
        subset_indices = indices[start:end]
        subset = Subset(trainset, subset_indices)
        train_loaders.append(DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=0))
            
    print("Loaders Ready.")
    return train_loaders, test_loader, global_train_loader

def get_ring_topology(n):
    """
    Returns a doubly stochastic adjacency matrix for a ring topology.
    W_ii = 1/3, W_{i, i-1} = 1/3, W_{i, i+1} = 1/3
    """
    if n == 1:
        return torch.ones(1, 1)
        
    W = torch.zeros(n, n)
    for i in range(n):
        W[i, i] = 1/3
        W[i, (i - 1) % n] = 1/3
        W[i, (i + 1) % n] = 1/3
        
    return W

def plot_results(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Plot Generalization Gap vs Iteration
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('Experiment'):
        plt.plot(group['Iteration'], group['Gap'], label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Generalization Gap')
    plt.title('Generalization Gap vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'gap_vs_iter.png'))
    plt.close()

    # Plot Test Acc vs Communication Bits
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('Experiment'):
        plt.plot(group['CommBits'], group['TestAcc'], label=name)
        
    plt.xlabel('Communication Bits')
    plt.ylabel('Test Accuracy')
    plt.title('Test Acc vs Communication Bits')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'acc_vs_bits.png'))
    plt.close()
