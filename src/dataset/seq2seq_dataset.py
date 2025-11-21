import numpy as np
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

class NextGestureDataset(Dataset):
    """
    Dataset that takes raw windows and splits into (x_in, y_out).
    """
    def __init__(self, windows, t_in=24, t_out=12,
                 sample_rate=48, lowpass_cutoff=15,
                 mean_removal=True, normalize=True, lowpass=True, smoothing=True):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.t_in = t_in
        self.t_out = t_out

        # Preprocessing flags
        self.mean_removal = mean_removal
        self.normalize = normalize
        self.lowpass = lowpass
        self.smoothing = smoothing

        # Low-pass filter setup
        if self.lowpass:
            self.b, self.a = butter(4, lowpass_cutoff / (sample_rate / 2), btype="low")

    def transform(self, window: np.ndarray) -> np.ndarray:

        # 3. Low-pass filtering
        if self.lowpass:
            window = filtfilt(self.b, self.a, window, axis=0, padlen=min(15, window.shape[0]-1))

        # 4. Moving average smoothing
        kernel = np.ones(5) / 5
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=window
        )
        return smoothed.astype(np.float32)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window = self.windows[idx].numpy()
        window = self.transform(window)
        window = torch.tensor(window, dtype=torch.float32)

        # Split into input and output
        x_in = window[:self.t_in]                      # [t_in, 6]
        y_out = window[self.t_in:self.t_in+self.t_out] # [t_out, 6]

        return x_in, y_out

    @staticmethod
    def combine_csv_files(pattern="*.csv", window_size=36, stride=6):
        """
        Build raw windows from CSV files. Each window will be split by the dataset.
        """
        all_windows = []
        for f in glob.glob(pattern):
            df = pd.read_csv(f)
            data = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values

            for start in range(0, len(data) - window_size + 1, stride):
                window = data[start:start+window_size]
                all_windows.append(window)

        all_windows = np.stack(all_windows)
        print("Final dataset shape (windows):", all_windows.shape)
        return all_windows
