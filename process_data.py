import os
import re
import h5py
import torch
import numpy as np

class PreprocessedScatteringDataset:
    """
    Vectorized processing of scattering data. Saves normalized tensors to disk.
    """

    def __init__(self, raw_data_dir=None, processed_dir=None):
        if raw_data_dir is None:
            raw_data_dir = os.path.join(os.environ["WORK"], "data_generate", "data/nips_prelim/volume3")
        if processed_dir is None:
            processed_dir = os.path.join(os.environ["WORK"], "data_generate", "processed_nips_prelim/volume3")
        os.makedirs(processed_dir, exist_ok=True)

        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.h5_pattern = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"

        # Gather all Hamiltonians to compute global min/max
        self._compute_hparam_min_max()

        # Process all H5 files
        self._process_all_files()

    def _compute_hparam_min_max(self):
        hparams_all = []
        for fname in os.listdir(self.raw_data_dir):
            if not fname.endswith(".h5"):
                continue
            match = re.search(self.h5_pattern, fname)
            if match:
                hparams_all.append([float(x) for x in match.groups()])
        hparams_all = torch.tensor(hparams_all, dtype=torch.float32)
        self.hparam_min = hparams_all.min(dim=0)[0]
        self.hparam_max = hparams_all.max(dim=0)[0]

    def _process_all_files(self):
        for fname in os.listdir(self.raw_data_dir):
            if not fname.endswith(".h5"):
                continue
            file_path = os.path.join(self.raw_data_dir, fname)
            print(f"Processing {fname} ...")
            self._process_file(file_path, fname)

    def _process_file(self, file_path, fname):
        match = re.search(self.h5_pattern, fname)
        if not match:
            print(f"Skipping {fname}, pattern not matched.")
            return
        hparams = torch.tensor([float(x) for x in match.groups()], dtype=torch.float32)

        with h5py.File(file_path, "r") as f:
            qa = f["qa_range"][:]
            qb = f["qb_range"][:]
            qc = f["qc_range"][:]
            energies = f["energies"][:]
            data = f["data"][:]

        nE, nA, nB, nC = len(energies), len(qa), len(qb), len(qc)
        data = data.reshape(nE, nA, nB, nC)

        # Vectorized coordinates
        E, QA, QB, QC = np.meshgrid(energies, qa, qb, qc, indexing='ij')
        coords = np.stack([QA, QB, QC, E], axis=-1).reshape(-1, 4)
        values = data.reshape(-1, 1)

        # Normalize coordinates
        coords[:,0] = 2*(coords[:,0] - qa.min())/(qa.max() - qa.min()) - 1
        coords[:,1] = 2*(coords[:,1] - qb.min())/(qb.max() - qb.min()) - 1
        coords[:,2] = 2*(coords[:,2] - qc.min())/(qc.max() - qc.min()) - 1
        coords[:,3] = 2*(coords[:,3] - energies.min())/(energies.max() - energies.min()) - 1

        # Normalize Hamiltonians
        hparams_norm = 2*(hparams - self.hparam_min) / (self.hparam_max - self.hparam_min)
        hparams_norm = hparams_norm.repeat(coords.shape[0], 1)

        # Concatenate
        x = np.concatenate([coords, hparams_norm.numpy()], axis=1)
        y = values

        # Convert to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Save tensors
        save_x = os.path.join(self.processed_dir, fname.replace(".h5", "_x.pt"))
        save_y = os.path.join(self.processed_dir, fname.replace(".h5", "_y.pt"))
        torch.save(x_tensor, save_x)
        torch.save(y_tensor, save_y)
        print(f"Saved {save_x} and {save_y} ({x_tensor.shape[0]} samples)")

if __name__ == "__main__":
    raw_dir = os.path.join(os.environ["WORK"], "data_generate", "data/nips_prelim/volume3")
    processed_dir = os.path.join(os.environ["WORK"], "data_generate", "processed_nips_prelim/volume3")