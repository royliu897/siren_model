import torch
from siren import Siren

#hyperparameters-- to be tuned
in_features = 13 #9 physics (Ax, Az, ...) plus 4 coordinates, could possibly differntiate between the 9 and 4 later?
hidden_features = 64 #number of neurons per hidden layer
hidden_layers = 3 #number of hidden SineLayers
out_features = 1 #intensity
omega_0 = 30.0

model = Siren(in_features, hidden_features, hidden_layers, out_features, omega_0)

#dummy input
x = torch.tensor([[-0.01, 0.21, -2.7, -2.0, 0.2, 0.2, 13.9, 13.9, -0.38, 0.5, 2.5, 0.0, 0.1]], dtype=torch.float32) #could experiment with like deepseek using less expensive dtype, such as float8
print("Input shape:", x.shape)

output = model(x)
print("Output shape:", output.shape)
print("Output:", output)
