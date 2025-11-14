import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from siren import HybridSiren
from visualize import save_axis_slices

# ---------------- Physics-aware and Focal Losses -----------------

def physics_log_loss(y_pred, y_true, eps=1e-6):
    """
    Physics-informed loss: L1 distance of log(1 + y)
    Emphasizes relative errors and suppresses extremely large magnitudes.
    """
    log_pred = torch.log1p(torch.relu(y_pred) + eps)
    log_true = torch.log1p(torch.relu(y_true) + eps)
    return torch.mean(torch.abs(log_pred - log_true))


def focal_regression_loss_debug(y_pred, y_true, alpha=4.0, gamma=2.0, eps=1e-8):
    """
    NOT USED FOR BACKPROP — ONLY PRINTED.
    """
    y_norm = y_true / (y_true.max() + eps)
    err = torch.abs(y_pred - y_true) + eps
    w = (1 + alpha * y_norm) ** gamma
    return torch.mean(w * err**2)


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


# ---------------- Config -----------------
PROCESSED_DIR = os.path.expandvars("$WORK/data_generate/processed_nips_prelim/volume3")
MODEL_DIR = os.path.expandvars("$WORK/data_generate/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Training -----------------
def train(x_tensor, y_tensor, save_model_path, epochs=100, batch_size=1024, lr=1e-4,
          visualize=True, orig_h5_file=None):

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = os.cpu_count()-1, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridSiren().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(epochs):
        running_phys_loss = 0.0

        # For printing metrics
        running_mae = 0.0
        running_focal = 0.0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Main physics-aware loss (used for training)
            loss = physics_log_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_phys_loss += loss.item() * x_batch.size(0)

            # Extra debugging metrics (not used for gradient)
            running_mae += mae_loss(y_pred, y_batch).item() * x_batch.size(0)
            running_focal += focal_regression_loss_debug(y_pred, y_batch).item() * x_batch.size(0)

        # Epoch metrics
        epoch_phys = running_phys_loss / len(dataset)
        epoch_mae = running_mae / len(dataset)
        epoch_focal = running_focal / len(dataset)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"PhysicsLogLoss={epoch_phys:.6f} | "
              f"MAE={epoch_mae:.6f} | "
              f"Focal(debug)={epoch_focal:.6f}")

        # Save only by physics-aware loss (main objective)
        if epoch_phys < best_loss:
            best_loss = epoch_phys
            torch.save(model.state_dict(), save_model_path)

    print(f"\nTraining complete. Best physics-aware loss: {best_loss:.6f}\n")

    # Visualization callback
    if visualize and orig_h5_file is not None:
        save_dir = os.path.join(os.path.dirname(save_model_path), "axis_slices",
                                os.path.splitext(os.path.basename(orig_h5_file))[0])
        save_axis_slices(model, orig_h5_file, energy_idx=50, save_dir=save_dir, device=device)

    return model


# ---------------- Main Loop -----------------
if __name__ == "__main__":
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_x.pt")]
    processed_files.sort()

    for x_file in processed_files:
        base_name = x_file.replace("_x.pt", "")
        y_file = x_file.replace("_x.pt", "_y.pt")

        x_tensor = torch.load(os.path.join(PROCESSED_DIR, x_file))
        y_tensor = torch.load(os.path.join(PROCESSED_DIR, y_file))

        model_save_path = os.path.join(MODEL_DIR, f"{base_name}_siren.pt")

        orig_h5_file = os.path.join(
            os.environ["WORK"],
            "data_generate/data/nips_prelim/volume3",
            f"{base_name}.h5"
        )

        train(x_tensor, y_tensor, save_model_path=model_save_path,
              epochs=100, batch_size=1024, orig_h5_file=orig_h5_file)
