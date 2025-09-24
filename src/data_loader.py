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
    def __init__(self, X, Y, t_in=127, sample_rate=48, lowpass_cutoff=15,
                 mean_removal=True, normalize=True, lowpass=True, smoothing=True):
        self.X = torch.tensor(X, dtype=torch.float32) 
        self.Y = torch.tensor(Y, dtype=torch.float32)
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
        y_out = x[1:self.t_in] 
        y_label = self.Y[idx]  

        return x_in, y_out, y_label
       
    @staticmethod
    def make_windows(data, labels, window_size=24, stride=6, no_class="no_gesture"):

        X_windows, y_windows = [], []
        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]
            window_labels = labels[start:end]
            gesture_labels = [lab for lab in window_labels if lab != no_class]
            # this is using majority vote right here
            if len(gesture_labels) > 0:
              label = gesture_labels[0] 
            else:
              label = no_class   

            X_windows.append(window)
            y_windows.append(label)

        return np.stack(X_windows), np.array(y_windows)
    
    @staticmethod
    def combine_csv_files(pattern="*.csv", window_size=24, stride=6):

        X_all, y_all = [], []

        for f in glob.glob(pattern):
            df = pd.read_csv(f)

            # Extract IMU channels and labels
            data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values
            labels = df["gesture"].values

            # Windowing
            X, y = IMUSeq2SeqDataset.make_windows(data, labels, window_size, stride)
            X_all.append(X)
            y_all.append(y)

        # Concatenate all files
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        classes = np.unique(y_all)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_onehot = np.zeros((len(y_all), len(classes)), dtype=np.float32)
        for i, label in enumerate(y_all):
          y_onehot[i, class_to_idx[label]] = 1.0
        print("Classes:", classes)
        print("Final dataset:", X_all.shape, y_onehot.shape)
        return X_all, y_onehot, classes

def split_train_val(X, Y, val_ratio=0.2, seed=42):
    """Split dataset into train/val sets"""
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return  X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
