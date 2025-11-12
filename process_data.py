import os
import re
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class ScatteringDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for fname in os.listdir(data_dir):
            if not fname.endswith(".h5"):
                continue

            # Extract Hamiltonian parameters from filename
            pattern = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"
            match = re.search(pattern, fname)
            if not match:
                continue
            Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4 = map(float, match.groups())
            hparams = torch.tensor([Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4])

            with h5py.File(os.path.join(data_dir, fname), "r") as f:
                qa = f["qa_range"][:]
                qb = f["qb_range"][:]
                qc = f["qc_range"][:]
                energies = f["energies"][:]
                data = f["data"][:]

            nE, nA, nB, nC = len(energies), len(qa), len(qb), len(qc)
            data = data.reshape(nE, nA, nB, nC)
            for iE, E in enumerate(energies):
                for iA, a in enumerate(qa):
                    for iB, b in enumerate(qb):
                        for iC, c in enumerate(qc):
                            coord = torch.tensor([a, b, c, E])
                            x = torch.cat([coord, hparams])  # 13 inputs total
                            y = torch.tensor([data[iE, iA, iB, iC]])
                            self.samples.append((x, y))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
