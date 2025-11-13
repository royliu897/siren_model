import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from siren import HybridSiren
from visualize import save_axis_slices

# ---------------- Config -----------------
PROCESSED_DIR = os.path.expandvars("$WORK/data_generate/processed_nips_prelim/volume3")
MODEL_DIR = os.path.expandvars("$WORK/data_generate/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Training -----------------
def train(x_tensor, y_tensor, save_model_path, epochs=100, batch_size=1024, lr=1e-4, visualize=True, orig_h5_file=None):
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridSiren().to(device)
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_model_path)

    print(f"Training complete. Best loss: {best_loss:.6f}")

    # Visualization
    if visualize and orig_h5_file is not None:
        save_dir = os.path.join(os.path.dirname(save_model_path), "axis_slices",
                                os.path.splitext(os.path.basename(orig_h5_file))[0])
        save_axis_slices(model, orig_h5_file, energy_idx=50, save_dir=save_dir, device=device)

    return model

# ---------------- Main Loop -----------------
if __name__ == "__main__":
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_x.pt")]
    processed_files.sort()  # optional: sort alphabetically

    for x_file in processed_files:
        base_name = x_file.replace("_x.pt", "")
        y_file = x_file.replace("_x.pt", "_y.pt")
        x_tensor = torch.load(os.path.join(PROCESSED_DIR, x_file))
        y_tensor = torch.load(os.path.join(PROCESSED_DIR, y_file))

        model_save_path = os.path.join(MODEL_DIR, f"{base_name}_siren.pt")
        # original h5 file for visualization
        orig_h5_file = os.path.join(os.environ["WORK"], "data_generate/data/nips_prelim/volume3", f"{base_name}.h5")
        
        train(x_tensor, y_tensor, save_model_path=model_save_path, epochs=100, batch_size=1024, orig_h5_file=orig_h5_file)
