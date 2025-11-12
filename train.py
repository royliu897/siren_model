import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from siren import Siren
from visualize import save_axis_slices
from process_data import StreamingScatteringDataset

# ---------------- Config -----------------
DATA_DIR = os.path.expandvars("$WORK/data_generate/data/nips_prelim/volume3")
MODEL_DIR = os.path.expandvars("$WORK/data_generate/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Training -----------------
def train(h5_file, save_model_path, epochs=100, batch_size=1024, lr=1e-4, visualize=True, sample_size=None):
    dataset = StreamingScatteringDataset(data_dir=os.path.dirname(h5_file), sample_size=sample_size)
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
        train(h5_file, save_model_path=model_save_path, epochs=100, sample_size=100_000)
