from siren import HybridSiren
import torch

model = HybridSiren()
x = torch.randn(8, 13)
y = model(x)
print(y.shape)  # should be [8, 1]
