import torch
print(torch.__version__)
import torch.nn.functional as F

weight_cpu = torch.randn(1, 4, 10, device="cpu")
weight_mps = weight_cpu.detach().clone().to("mps")

nc = 65536 # OK
nc = 66000 # NotImplementedError: Output channels > 65536 not supported at the MPS device.
x_cpu = torch.randn(1, 4, nc, device="cpu")
x_mps = x_cpu.detach().clone().to("mps")

y_cpu = F.conv1d(x_cpu, weight_cpu)
y_mps = F.conv1d(x_mps, weight_mps)

print(y_cpu)
print(y_mps)