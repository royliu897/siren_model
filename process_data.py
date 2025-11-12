import os
import re
import h5py
import torch
from torch.utils.data import Dataset

class StreamingScatteringDataset(Dataset):
    def __init__(self, data_dir=None, sample_size=None):
        """
        data_dir: folder with h5 files
        sample_size: optional, randomly sample N points per file
        """
        if data_dir is None:
            data_dir = os.path.join(os.environ["WORK"], "data_generate", "data", "nips_prelim", "volume3")

        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
        self.h5_pattern = r"Ax=(-?\d+\.\d+)_Az=(-?\d+\.\d+)_J1a=(-?\d+\.\d+)_J1b=(-?\d+\.\d+)_J2a=(-?\d+\.\d+)_J2b=(-?\d+\.\d+)_J3a=(-?\d+\.\d+)_J3b=(-?\d+\.\d+)_J4=(-?\d+\.\d+)"
        self.sample_size = sample_size

        # Precompute Hamiltonian min/max across files for normalization
        hparams_all = []
        for file in self.data_files:
            fname = os.path.basename(file)
            match = re.search(self.h5_pattern, fname)
            if match:
                hparams = [float(x) for x in match.groups()]
                hparams_all.append(hparams)
        hparams_all = torch.tensor(hparams_all, dtype=torch.float32)
        self.hparam_min = hparams_all.min(dim=0)[0]
        self.hparam_max = hparams_all.max(dim=0)[0]

        # For indexing samples across all files
        self.file_indices = []
        for i, file in enumerate(self.data_files):
            with h5py.File(file, "r") as f:
                nE, nA, nB, nC = len(f["energies"]), len(f["qa_range"]), len(f["qb_range"]), len(f["qc_range"])
                total_points = nE * nA * nB * nC
                if self.sample_size and self.sample_size < total_points:
                    indices = torch.randperm(total_points)[:self.sample_size]
                else:
                    indices = torch.arange(total_points)
                for idx in indices:
                    self.file_indices.append((i, int(idx)))

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, point_idx = self.file_indices[idx]
        file_path = self.data_files[file_idx]

        # Open H5 and read shapes
        with h5py.File(file_path, "r") as f:
            qa = f["qa_range"][:]
            qb = f["qb_range"][:]
            qc = f["qc_range"][:]
            energies = f["energies"][:]
            data = f["data"][:].reshape(len(energies), len(qa), len(qb), len(qc))

            # Convert flat index to multi-dimensional indices
            nE, nA, nB, nC = len(energies), len(qa), len(qb), len(qc)
            iE = point_idx // (nA * nB * nC)
            rem = point_idx % (nA * nB * nC)
            iA = rem // (nB * nC)
            rem = rem % (nB * nC)
            iB = rem // nC
            iC = rem % nC

            a, b, c, E = qa[iA], qb[iB], qc[iC], energies[iE]
            y = data[iE, iA, iB, iC]

        # Normalize coordinates
        a_norm = 2 * (a - qa.min()) / (qa.max() - qa.min()) - 1
        b_norm = 2 * (b - qb.min()) / (qb.max() - qb.min()) - 1
        c_norm = 2 * (c - qc.min()) / (qc.max() - qc.min()) - 1
        E_norm = 2 * (E - energies.min()) / (energies.max() - energies.min()) - 1

        # Normalize Hamiltonian
        fname = os.path.basename(file_path)
        match = re.search(self.h5_pattern, fname)
        hparams = torch.tensor([float(x) for x in match.groups()], dtype=torch.float32)
        hparams_norm = 2 * (hparams - self.hparam_min) / (self.hparam_max - self.hparam_min) - 1

        x = torch.tensor([a_norm, b_norm, c_norm, E_norm], dtype=torch.float32)
        x = torch.cat([x, hparams_norm])

        return x, torch.tensor([y], dtype=torch.float32)
