import torch
from models import ResNet20

try:
    print("Creating model...")
    net = ResNet20().cuda()
    print("Model created.")
    x = torch.randn(1, 3, 32, 32).cuda()
    y = net(x)
    print("Forward pass done.")
except Exception as e:
    print(f"Error: {e}")
