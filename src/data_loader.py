# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

class MUSeq2SeqDataset(Dataset):
    """
    Preprocess raw IMU window (mean removal, normalization, filtering, smoothing).
    """
    def __init__(self, X, t_in=127, sample_rate=48, lowpass_cutoff=15,
                 mean_removal=True, normalize=True, lowpass=True, smoothing=True):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T, C)
        self.t_in = t_in

        # Flags for preprocessing
        self.mean_removal = mean_removal
        self.normalize = normalize
        self.lowpass = lowpass
        self.smoothing = smoothing

        # Low-pass filter setup
        if self.lowpass:
            self.b, self.a = butter(4, lowpass_cutoff / (sample_rate / 2), btype="low")

    def transform(self, window: np.ndarray) -> np.ndarray:
        # 1. Mean removal
        window = window - np.mean(window, axis=0, keepdims=True)

        # 2. Normalize to [-1,1]
        denom = np.maximum(np.max(np.abs(window), axis=0, keepdims=True), 1e-6)
        window = window / denom

        # 3. filtering
        window = filtfilt(self.b, self.a, window, axis=0)

        # 4. Moving average
        kernel = np.ones(5) / 5
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=window
        )
        return smoothed.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert tensor to numpy for preprocessing
        x = self.X[idx].numpy()           
        x = self.transform(x)           
        x = torch.tensor(x, dtype=torch.float32)

        x_in = x[:self.t_in]             
        y_out = x[1:self.t_in+1]          
        return x_in, y_out


def split_train_val(X, val_ratio=0.2, seed=42):
    """Split dataset into train/val sets"""
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx]
