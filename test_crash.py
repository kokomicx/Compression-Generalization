import torch
print("Torch imported")
try:
    from compressor import QSGDCompressor
    print("Compressor imported")
    c = QSGDCompressor(8)
    print("Compressor instantiated")
    t = torch.randn(100).cuda()
    print("Tensor created")
    z = c.compress(t)
    print("Compression done")
except Exception as e:
    print(f"Error: {e}")
