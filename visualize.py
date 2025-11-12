import os
import re
import torch
import numpy as np
import h5py
import plotly.graph_objects as go

pattern = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"

def save_axis_slices(model, file_path, energy_idx=50, save_dir="axis_slices", device=None):
    os.makedirs(save_dir, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load H5
    with h5py.File(file_path, "r") as f:
        qa = f["qa_range"][:]
        qb = f["qb_range"][:]
        qc = f["qc_range"][:]
        energies = f["energies"][:]
        data = f["data"][:].reshape(len(energies), len(qa), len(qb), len(qc))

    # Parse Hamiltonian params
    fname = os.path.basename(file_path)
    match = re.search(pattern, fname)
    if not match:
        raise ValueError("Filename doesn't match expected pattern")
    Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4 = map(float, match.groups())

    # Energy slice
    E = energies[energy_idx]
    gt_slice = data[energy_idx]  # shape: (len(qa), len(qb), len(qc))

    # Normalize coordinates and Hamiltonians (match training)
    qa_norm = qa  # optional: normalize if you normalized in training
    qb_norm = qb
    qc_norm = qc
    E_norm = E  # optional: normalize energy
    hparams = torch.tensor([Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4], dtype=torch.float32)
    hparams = hparams.to(device)

    # Prepare coordinate grid for batch prediction
    QA, QB, QC = np.meshgrid(qa_norm, qb_norm, qc_norm, indexing="ij")
    QA = QA.ravel()
    QB = QB.ravel()
    QC = QC.ravel()
    E_arr = np.full_like(QA, E_norm)

    coords = torch.tensor(np.stack([QA, QB, QC, E_arr], axis=1), dtype=torch.float32).to(device)
    hparams_exp = hparams.expand(coords.size(0), -1)
    inputs = torch.cat([coords, hparams_exp], dim=1)

    # Batched prediction to avoid memory issues
    batch_size = 16_384  # adjust for your GPU/CPU memory
    preds = []
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            end = start + batch_size
            batch = inputs[start:end]
            preds.append(model(batch).cpu())
    pred_slice = torch.cat(preds, dim=0).numpy().reshape(len(qa), len(qb), len(qc))

    # Axis slices
    slices = {
        "XY": gt_slice[:, :, len(qc)//2],
        "XZ": gt_slice[:, len(qb)//2, :],
        "YZ": gt_slice[len(qa)//2, :, :]
    }
    pred_slices = {
        "XY": pred_slice[:, :, len(qc)//2],
        "XZ": pred_slice[:, len(qb)//2, :],
        "YZ": pred_slice[len(qa)//2, :, :]
    }

    # Save HTML heatmaps
    for axis in slices:
        # Ground truth
        fig = go.Figure(data=go.Heatmap(z=slices[axis], colorscale="Viridis"))
        fig.update_layout(title=f"Ground Truth {axis}-slice at E={E:.2f}")
        fig.write_html(os.path.join(save_dir, f"gt_{axis}.html"))

        # Predicted
        fig = go.Figure(data=go.Heatmap(z=pred_slices[axis], colorscale="Viridis"))
        fig.update_layout(title=f"Predicted {axis}-slice at E={E:.2f}")
        fig.write_html(os.path.join(save_dir, f"pred_{axis}.html"))

    print(f"Saved axis slices to {save_dir}")
