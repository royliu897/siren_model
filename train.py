import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from siren import HybridSiren
from visualize import save_axis_slices

# ---------------- Loss functions -----------------
def log_mse_loss(y_pred, y_true, eps=1e-6):
    """Log-scaled MSE for better peak representation."""
    log_pred = torch.log1p(torch.relu(y_pred) + eps)
    log_true = torch.log1p(torch.relu(y_true) + eps)
    return torch.mean((log_pred - log_true) ** 2)

def second_derivative_loss(y_pred, coords, axis_dim=-1):
    """
    Encourage sharp peak fitting by penalizing curvature.
    Computes finite difference along the last axis (typically energy).
    coords: [N,4] tensor of (QA,QB,QC,E)
    axis_dim: dimension along which to compute second derivative
    """
    # Assuming data is sorted along axis_dim
    # approximate d2y/dx2 ~ y[i+1] - 2*y[i] + y[i-1]
    y = y_pred.view(-1)
    d2y = y[2:] - 2 * y[1:-1] + y[:-2]
    return torch.mean(d2y ** 2)

def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


# ---------------- Config -----------------
PROCESSED_DIR = os.path.expandvars("$WORK/data_generate/processed_nips_prelim/volume3")
MODEL_DIR = os.path.expandvars("$WORK/data_generate/models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- Training -----------------
def train(x_tensor, y_tensor, save_model_path, epochs=100, batch_size=1024,
          lr=1e-4, visualize=True, orig_h5_file=None, curvature_weight=1.0):

    dataset = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(16, os.cpu_count() - 1),
        pin_memory=True,
        persistent_workers=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridSiren().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        running_loss = torch.tensor(0.0, device=device)
        running_mae  = torch.tensor(0.0, device=device)
        running_log_debug = torch.tensor(0.0, device=device)

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(x_batch)

            # Log-MSE primary loss
            log_mse = log_mse_loss(y_pred, y_batch)

            # Curvature loss (approximate along energy axis, axis=-1 in input coords)
            curvature = second_derivative_loss(y_pred, x_batch[:,3], axis_dim=-1)

            # Total loss
            loss = log_mse + curvature_weight * curvature
            loss.backward()
            optimizer.step()

            # Debug metrics
            with torch.no_grad():
                running_loss += loss.detach() * x_batch.size(0)
                running_mae += mae_loss(y_pred, y_batch) * x_batch.size(0)
                running_log_debug += log_mse * x_batch.size(0)

        epoch_loss = (running_loss / len(dataset)).item()
        epoch_mae  = (running_mae / len(dataset)).item()
        epoch_log  = (running_log_debug / len(dataset)).item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"TotalLoss={epoch_loss:.6f} | "
              f"MAE(debug)={epoch_mae:.6f} | "
              f"LogMSE(debug)={epoch_log:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_model_path)

    print(f"\nTraining complete. Best total loss: {best_loss:.6f}\n")

    # Visualization
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
              epochs=100, batch_size=1024, orig_h5_file=orig_h5_file,
              curvature_weight=1.0)
