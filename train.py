import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from siren_model import Siren
from visualize_slices import save_axis_slices
import h5py
import numpy as np

# ---------------- Config -----------------
DATA_DIR = "/path/to/data/volume3"
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
H5_PATTERN = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"

# ---------------- Dataset -----------------
class H5SpinDataset(Dataset):
    def __init__(self, file_path, sample_size=100_000, freeze_hamiltonian=True):
        with h5py.File(file_path, 'r') as f:
            qa = np.array(f['qa_range'])
            qb = np.array(f['qb_range'])
            qc = np.array(f['qc_range'])
            energies = np.array(f['energies'])
            data = np.array(f['data'])

        data = data.reshape(len(energies), len(qa), len(qb), len(qc))
        E, QA, QB, QC = np.meshgrid(energies, qa, qb, qc, indexing='ij')
        coords = np.stack([QA, QB, QC, E], axis=-1).reshape(-1, 4)
        values = data.reshape(-1, 1)

        if freeze_hamiltonian:
            fname = os.path.basename(file_path)
            match = re.search(H5_PATTERN, fname)
            if not match:
                raise ValueError(f"Filename doesn't match expected pattern: {fname}")
            Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4 = map(float, match.groups())
            hparams = np.array([Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4], dtype=np.float32)
            hparams = np.repeat(hparams[None, :], len(coords), axis=0)
            coords = np.concatenate([coords, hparams], axis=1)

        if sample_size < len(coords):
            idx = np.random.choice(len(coords), sample_size, replace=False)
            coords = coords[idx]
            values = values[idx]

        self.x = torch.tensor(coords, dtype=torch.float32)
        self.y = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------- Training -----------------
def train_siren(h5_file, save_model_path, epochs=100, batch_size=1024, lr=1e-4, visualize=True):
    dataset = H5SpinDataset(h5_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Siren(in_features=13, hidden_features=256, hidden_layers=4, out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[{os.path.basename(h5_file)}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_model_path)

    print(f"Training complete for {os.path.basename(h5_file)}. Best loss: {best_loss:.6f}")

    if visualize:
        save_dir = os.path.join(os.path.dirname(save_model_path), "axis_slices", os.path.splitext(os.path.basename(h5_file))[0])
        save_axis_slices(model, h5_file, energy_idx=50, save_dir=save_dir, device=device)

    return model

# ---------------- Main Loop -----------------
if __name__ == "__main__":
    h5_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".h5")]

    for h5_file in h5_files:
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        model_save_path = os.path.join(MODEL_DIR, f"{base_name}_siren.pt")
        train_siren(h5_file, save_model_path=model_save_path, epochs=100)
