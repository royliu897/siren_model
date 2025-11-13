import os
import re
import h5py
import torch
from torch.utils.data import Dataset

class ScatteringDataset(Dataset):
    def __init__(self, data_dir=None, sample_size=None):
        """
        data_dir: folder with h5 files
        sample_size: optional, randomly sample N points per file
        """
        if data_dir is None:
            data_dir = os.path.join(os.environ["WORK"], "data_generate", "data", "nips_prelim", "volume3")

        self.samples = []
        self.h5_pattern = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"

        # Gather all hparams across files to compute global min/max
        hparams_all = []

        h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
        for fname in h5_files:
            match = re.search(self.h5_pattern, fname)
            if match:
                hparams_all.append([float(x) for x in match.groups()])
        hparams_all = torch.tensor(hparams_all, dtype=torch.float32)
        hparam_min = hparams_all.min(dim=0)[0]
        hparam_max = hparams_all.max(dim=0)[0]

        # Process each H5 file
        for fname in h5_files:
            file_path = os.path.join(data_dir, fname)
            match = re.search(self.h5_pattern, fname)
            if not match:
                continue
            hparams = torch.tensor([float(x) for x in match.groups()], dtype=torch.float32)

            with h5py.File(file_path, "r") as f:
                qa = f["qa_range"][:]
                qb = f["qb_range"][:]
                qc = f["qc_range"][:]
                energies = f["energies"][:]
                data = f["data"][:]

            nE, nA, nB, nC = len(energies), len(qa), len(qb), len(qc)
            data = data.reshape(nE, nA, nB, nC)

            # Compute min/max for coordinates and energies
            qa_min, qa_max = qa.min(), qa.max()
            qb_min, qb_max = qb.min(), qb.max()
            qc_min, qc_max = qc.min(), qc.max()
            E_min, E_max = energies.min(), energies.max()

            for iE, E in enumerate(energies):
                for iA, a in enumerate(qa):
                    for iB, b in enumerate(qb):
                        for iC, c in enumerate(qc):
                            # Normalize coordinates
                            a_norm = 2*(a - qa_min)/(qa_max - qa_min) - 1
                            b_norm = 2*(b - qb_min)/(qb_max - qb_min) - 1
                            c_norm = 2*(c - qc_min)/(qc_max - qc_min) - 1
                            E_norm = 2*(E - E_min)/(E_max - E_min) - 1

                            # Normalize Hamiltonian
                            hparams_norm = 2*(hparams - hparam_min)/(hparam_max - hparam_min) - 1

                            # Concatenate normalized inputs
                            x = torch.cat([torch.tensor([a_norm, b_norm, c_norm, E_norm], dtype=torch.float32),
                                           hparams_norm])
                            y = torch.tensor([data[iE, iA, iB, iC]], dtype=torch.float32)
                            self.samples.append((x, y))

        # Optionally subsample points
        if sample_size is not None and sample_size < len(self.samples):
            indices = torch.randperm(len(self.samples))[:sample_size]
            self.samples = [self.samples[i] for i in indices]

        print(f"Loaded {len(self.samples)} normalized samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
