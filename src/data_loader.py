# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset

class IMUSeq2SeqDataset(Dataset):
    """
    Dataset for IMU sequence-to-sequence task
      - input  = X[:T_in]
      - target = X[1:T_in+1]
    """
    def __init__(self, X, t_in=127, mean_std=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.t_in = t_in

        # normalization
        if mean_std is None:
            mu = self.X.mean(dim=(0, 1), keepdim=True)  # (1,1,C)
            sd = self.X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
        else:
            mu, sd = mean_std
        self.X = (self.X - mu) / sd
        self.mu, self.sd = mu, sd

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]              # (T=128, C)
        x_in  = x[:self.t_in]        # (T_in, C)
        y_out = x[1:self.t_in+1]     # (T_in, C)
        return x_in, y_out


def split_train_val(X, val_ratio=0.2, seed=42):
    """Split dataset into train/val sets"""
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx]
