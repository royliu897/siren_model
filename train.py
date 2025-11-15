import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from siren import Siren

# ------------------------------
# Dataset
# ------------------------------
class NIPSDataset(Dataset):
    def __init__(self, h5_file, scale_min=0.0, scale_max=0.5):
        self.data = h5py.File(h5_file, 'r')['data'][:]  # shape: (num_points, 9)
        
        # Separate features
        self.hamiltonian = self.data[:, :4]      # Ax, Az, J1a, J1b
        self.momentum = self.data[:, 4:8]        # h, k, l, w
        self.S = self.data[:, 8:]                # scattering intensity

        # Scale to [scale_min, scale_max]
        self.hamiltonian_scaled = self.minmax_scale(self.hamiltonian, scale_min, scale_max)
        self.momentum_scaled = self.minmax_scale(self.momentum, scale_min, scale_max)

        # Concatenate to final input: 8 features
        self.x = np.concatenate([self.hamiltonian_scaled, self.momentum_scaled], axis=1).astype(np.float32)
        self.y = self.S.astype(np.float32)

    @staticmethod
    def minmax_scale(arr, min_val, max_val):
        arr_min = arr.min(axis=0, keepdims=True)
        arr_max = arr.max(axis=0, keepdims=True)
        scaled = (arr - arr_min) / (arr_max - arr_min + 1e-12)  # avoid div by zero
        return scaled * (max_val - min_val) + min_val

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ------------------------------
# Hyperparameters
# ------------------------------
H5_FILE = os.path.join(os.environ["WORK"], "data_generate", "4d_data", "nips_100h_200pts.h5")
BATCH_SIZE = 1024
EPOCHS = 40
LEARNING_RATE = 1e-4
HIDDEN_FEATURES = 256
HIDDEN_LAYERS = 3
OMEGA_0 = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Data
# ------------------------------
dataset = NIPSDataset(H5_FILE)
n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ------------------------------
# Model
# ------------------------------
model = Siren(
    in_features=8,          # 4 Hamiltonian + 4 momentum
    hidden_features=HIDDEN_FEATURES,
    hidden_layers=HIDDEN_LAYERS,
    out_features=1,         # S intensity
    omega_0=OMEGA_0
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.detach() * x_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1:03d}: Train Loss={train_loss.item():.6f}, Val Loss={val_loss.item():.6f}")

# ------------------------------
# Save model
# ------------------------------
torch.save(model.state_dict(), "siren_nips_model.pt")
print("Training complete. Model saved to siren_nips_model.pt")
