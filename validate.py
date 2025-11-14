import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from siren import HybridSiren  # your model definition

# ---------------- Config -----------------
PROCESSED_DIR = os.path.expandvars("$WORK/data_generate/processed_validation")
MODEL_PATH = os.path.expandvars("$WORK/data_generate/models/Ax=-0.00501_Az=0.22663_J1a=-0.05225_J1b=-2.19099_J2a=0.38684_J2b=0.07684_J3a=5.34049_J3b=1.49999_J4=-0.6492_siren.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024

# ---------------- Loss Functions -----------------
def compute_metrics(y_pred, y_true):
    with torch.no_grad():
        diff = y_pred - y_true
        mse = torch.mean(diff**2).item()
        rmse = torch.sqrt(torch.mean(diff**2)).item()
        mae = torch.mean(torch.abs(diff)).item()
        max_err = torch.max(torch.abs(diff)).item()
    return mse, rmse, mae, max_err

# ---------------- Main Testing -----------------
def test_model():
    print(f"Using device: {DEVICE}")
    
    # Load model
    model = HybridSiren().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_x.pt")]
    processed_files.sort()

    all_metrics = []

    for x_file in processed_files:
        base_name = x_file.replace("_x.pt","")
        y_file = x_file.replace("_x.pt","_y.pt")
        x_tensor = torch.load(os.path.join(PROCESSED_DIR, x_file))
        y_tensor = torch.load(os.path.join(PROCESSED_DIR, y_file))

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        y_preds = []
        y_trues = []

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                y_pred = model(xb)
                y_preds.append(y_pred)
                y_trues.append(yb)

        y_pred_full = torch.cat(y_preds, dim=0)
        y_true_full = torch.cat(y_trues, dim=0)

        mse, rmse, mae, max_err = compute_metrics(y_pred_full, y_true_full)
        print(f"{base_name}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MaxErr={max_err:.6f}")
        all_metrics.append((mse, rmse, mae, max_err))

    # Average metrics across all validation files
    avg_metrics = torch.tensor(all_metrics).mean(dim=0)
    print(f"\nAverage metrics across validation set:")
    print(f"MSE={avg_metrics[0]:.6f}, RMSE={avg_metrics[1]:.6f}, MAE={avg_metrics[2]:.6f}, MaxErr={avg_metrics[3]:.6f}")

if __name__ == "__main__":
    test_model()
