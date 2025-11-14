import torch
import torch.nn as nn
import math

# ----- SIREN block -----
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# ----- Hybrid Model -----
class HybridSiren(nn.Module):
    def __init__(self, coord_dim=4, hparam_dim=9,
                 coord_hidden=128, hparam_hidden=64,
                 fusion_hidden=256, hidden_layers=3,
                 omega_0=30):
        super().__init__()

        # --- Coordinate branch (SIREN) ---
        coord_layers = [SineLayer(coord_dim, coord_hidden, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers - 1):
            coord_layers.append(SineLayer(coord_hidden, coord_hidden, omega_0=omega_0))
        self.coord_net = nn.Sequential(*coord_layers)

        # --- Hamiltonian branch (MLP) ---
        self.hparam_net = nn.Sequential(
            nn.Linear(hparam_dim, hparam_hidden),
            nn.Tanh(),
            nn.Linear(hparam_hidden, hparam_hidden),
            nn.Tanh()
        )

        # --- Fusion head ---
        self.fusion = nn.Sequential(
            nn.Linear(coord_hidden + hparam_hidden, fusion_hidden),
            nn.Tanh(),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.Tanh(),
            nn.Linear(fusion_hidden, 1)
        )

    def forward(self, x):
        coords = x[:, :4]     # (a,b,c,E)
        hparams = x[:, 4:]    # Hamiltonian

        coord_feat = self.coord_net(coords)
        hparam_feat = self.hparam_net(hparams)

        fused = torch.cat([coord_feat, hparam_feat], dim=-1)
        return self.fusion(fused)
