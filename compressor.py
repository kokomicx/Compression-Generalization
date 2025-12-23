import torch
import math

class Compressor:
    def compress(self, tensor):
        raise NotImplementedError

class IdentityCompressor(Compressor):
    def compress(self, tensor):
        return tensor.clone().detach()

class TopKCompressor(Compressor):
    def __init__(self, ratio):
        self.ratio = ratio

    def compress(self, tensor):
        numel = tensor.numel()
        k = max(1, int(numel * self.ratio))
        
        # Flatten tensor
        flattened = tensor.view(-1)
        abs_flattened = flattened.abs()
        
        # Find top k indices
        _, indices = torch.topk(abs_flattened, k)
        
        # Create mask or sparse tensor logic, but simpler to return zeroed out dense tensor for simulation
        compressed = torch.zeros_like(flattened)
        compressed[indices] = flattened[indices]
        
        return compressed.view_as(tensor).detach()

class QSGDCompressor(Compressor):
    def __init__(self, num_levels, bucket_size=128):
        self.num_levels = num_levels
        self.bucket_size = bucket_size

    def compress(self, tensor):
        # Flatten
        original_shape = tensor.shape
        flat = tensor.view(-1)
        numel = flat.numel()
        
        # Pad if needed
        pad_len = (self.bucket_size - (numel % self.bucket_size)) % self.bucket_size
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        
        num_buckets = flat.numel() // self.bucket_size
        reshaped = flat.view(num_buckets, self.bucket_size)
        
        norms = reshaped.norm(dim=1, keepdim=True)
        # Handle zero norms
        norms[norms == 0] = 1e-8 # Avoid div by zero, effectively zeroing out those buckets
        
        abs_tensor = reshaped.abs()
        level_float = self.num_levels * (abs_tensor / norms)
        
        floor_level = level_float.floor()
        prob = level_float - floor_level
        
        # Safety
        prob = torch.clamp(prob, min=0.0, max=1.0)
        mask = (torch.rand_like(prob) < prob).float()
        
        quantized = floor_level + mask
        
        scale = norms / self.num_levels
        compressed = scale * quantized * reshaped.sign()
        
        # Zero out buckets that had zero norm originally (or close to it)
        # Though math handles it (scale approx 0), explicit check safer? 
        # No, 1e-8 scale is fine.
        
        compressed = compressed.view(-1)
        if pad_len > 0:
            compressed = compressed[:-pad_len]
            
        return compressed.view(original_shape).detach()
