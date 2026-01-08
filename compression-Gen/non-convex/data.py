# non-convex/data.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from config import *

def get_cifar10_dataloaders():
    print("Preparing CIFAR-10 Data...")
    
    # 标准数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # IID Partitioning
    num_train = len(trainset)
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    
    # Split indices for each agent
    agent_indices = np.array_split(indices, NUM_AGENTS)
    
    train_loaders = []
    for i in range(NUM_AGENTS):
        subset = Subset(trainset, agent_indices[i])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        train_loaders.append(loader)
    
    # Global Test Loader (Used for evaluation)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loaders, test_loader