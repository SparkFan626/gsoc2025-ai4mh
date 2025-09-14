import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.t()
    torch.cuda.synchronize()
    print("CUDA kernel OK on:", torch.cuda.get_device_name(0), y.shape)
except Exception as e:
    print("CUDA kernel FAILED:", repr(e))
