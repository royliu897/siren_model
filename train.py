import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from siren import Siren
from visualize import save_axis_slices
import h5py
import numpy as np
from tqdm import tqdm

# ---------------- Config -----------------
DATA_DIR = os.path.expandvars("$WORK/data_generate/data/4d_only")
MODEL_DIR = os.path.expandvars("$WORK/data_generate/4d_models")
os.makedirs(MODEL_DIR, exist_ok=True)

H5_PATTERN = r"J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)"

# ---------------- Dataset -----------------
class H5SpinDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            qa = np.array(f['qa'])   # was 'qa_range'
            qb = np.array(f['qb'])   # was 'qb_range'
            qc = np.array(f['qc'])   # was 'qc_range'
            energies = np.array(f['energies'])
            data = np.array(f['data'])

        # reshape data
        data = data.reshape(len(energies), len(qa), len(qb), len(qc))
        E, QA, QB, QC = np.meshgrid(energies, qa, qb, qc, indexing='ij')
        coords = np.stack([QA, QB, QC, E], axis=-1).reshape(-1, 4)
        values = data.reshape(-1, 1)

        # normalize coordinates to [0,1]
        coords = np.stack([
            (QA - QA.min()) / (QA.max() - QA.min()),
            (QB - QB.min()) / (QB.max() - QB.min()),
            (QC - QC.min()) / (QC.max() - QC.min()),
            (E - E.min()) / (E.max() - E.min())
        ], axis=-1).reshape(-1, 4)

        # extract varying Hamiltonians from filename
        fname = os.path.basename(file_path)
        match = re.search(H5_PATTERN, fname)
        if not match:
            raise ValueError(f"Filename doesn't match expected pattern: {fname}")
        J1a, J1b, J3a, J3b = map(float, match.groups())
        hparams = np.array([J1a, J1b, J3a, J3b], dtype=np.float32)
        hparams /= np.max(np.abs(hparams))  # normalize Hamiltonians
        hparams = np.repeat(hparams[None, :], len(coords), axis=0)

        # concatenate coordinates + Hamiltonians
        coords = np.concatenate([coords, hparams], axis=1)  # 4 coords + 4 Hamiltonians = 8D

        # log-scale targets
        values = np.log1p(values)

        self.x = torch.tensor(coords, dtype=torch.float32)
        self.y = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------- Training -----------------
def train(h5_file, save_model_path, epochs=100, batch_size=65536, val_split=0.1, lr=1e-4, visualize=True):
    dataset = H5SpinDataset(h5_file)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size//2, shuffle=False, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Siren(in_features=8, hidden_features=256, hidden_layers=4, out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # mixed precision
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x_batch.size(0)
            pbar.set_postfix({'batch_loss': loss.item()})

        scheduler.step()
        train_loss = running_loss / train_len

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    y_pred = model(x_val)
                    val_loss += criterion(y_pred, y_val).item() * x_val.size(0)
        val_loss /= val_len
        model.train()

        print(f"[{os.path.basename(h5_file)}] Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)

    print(f"Training complete for {os.path.basename(h5_file)}. Best Val Loss: {best_val_loss:.6f}")

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
        train(h5_file, save_model_path=model_save_path, epochs=100)
