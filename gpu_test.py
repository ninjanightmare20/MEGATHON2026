import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(2000, 2000, device="cuda", dtype=torch.float16)
print("matmul ok:", (x @ x).float().sum().item())