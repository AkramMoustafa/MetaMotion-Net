# data_loader.py
import numpy as np
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

class IMUSeq2SeqDataset(Dataset):
    """
    Preprocess raw IMU window (mean removal, normalization, filtering, smoothing).
    """
    def __init__(self, X, Y, t_in=None, sample_rate=48, lowpass_cutoff=15,
                 mean_removal=True, normalize=True, lowpass=True, smoothing=True):
        self.X = torch.tensor(X, dtype=torch.float32) 
        self.Y = torch.tensor(Y, dtype=torch.float32) 
        self.t_in = t_in if t_in is not None else X.shape[1] // 2

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
        window = filtfilt(self.b, self.a, window, axis=0, padlen=min(15, window.shape[0]-1))

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
        y_out = x[self.t_in:]  

        return x_in, y_out
       
    @staticmethod
    def make_windows(data, window_size=24, stride=6, t_in=None, t_out=None):
        """
        Create sliding windows for sequence prediction.
        Each window is split into input (x_in) and output (y_out).
        """
        X_in, Y_out = [], []

        if t_in is None:
            t_in = window_size // 2   # default: half input
        if t_out is None:
            t_out = window_size - t_in

        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]

            # Split into input/output sequences
            x_in = window[:t_in]
            y_out = window[t_in:t_in + t_out]

            X_in.append(x_in)
            Y_out.append(y_out)

        return np.stack(X_in), np.stack(Y_out)

    @staticmethod
    def combine_csv_files(pattern="*.csv", window_size=24, stride=6, t_in=None, t_out=None):
        """
        Load multiple CSV files, build (X_in, Y_out) windows for seq2seq prediction.
        """
        X_all, Y_all = [], []

        for f in glob.glob(pattern):
            df = pd.read_csv(f)

            # Extract IMU channels only (no gesture column here)
            data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values

            # Build input/output windows
            X, Y = IMUSeq2SeqDataset.make_windows(data, window_size, stride, t_in, t_out)
            X_all.append(X)
            Y_all.append(Y)

        # Concatenate all files
        X_all = np.concatenate(X_all, axis=0)
        Y_all = np.concatenate(Y_all, axis=0)

        print("Final dataset:", X_all.shape, Y_all.shape)  
        return X_all, Y_all

    @staticmethod
    def split_train_val(X, Y, val_ratio=0.2, seed=42):
        """Split dataset into train/val sets"""
        np.random.seed(seed)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        n_val = int(len(X) * val_ratio)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return  X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
