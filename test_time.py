import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(10000, 1000).to(device)
y = torch.randn(1000, 10000).to(device)

start = time.time()
z = torch.matmul(x, y)
print(f"GPU: {time.time() - start:.4f}s")

x_cpu, y_cpu = x.cpu(), y.cpu()
start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
print(f"CPU: {time.time() - start:.4f}s")
