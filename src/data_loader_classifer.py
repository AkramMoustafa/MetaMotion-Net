from src.data_loader import IMUSeq2SeqDataset

import torch
import pandas as pd
import numpy as np

from src.data_loader import IMUSeq2SeqDataset

class IMUSeq2SeqClassifierDataset(IMUSeq2SeqDataset):
    """
    Same as IMUSeq2SeqDataset but returns gesture labels too.
    """
    def __init__(self, windows, window_labels, t_in=24, t_out=12, **kwargs):
        super().__init__(windows, t_in=t_in, t_out=t_out, **kwargs)
        assert len(window_labels) == len(windows), "window_labels length must match number of windows"
        self.window_labels = torch.tensor(window_labels, dtype=torch.long)

    def __getitem__(self, idx):
        x_in, y_out = super().__getitem__(idx)
        label = self.window_labels[idx]
        return x_in, y_out, label

    @staticmethod
    def build_window_labels(csv_path, window_size, stride, gesture_to_idx):
        """Build windows + per-window gesture labels from a CSV file."""
        df = pd.read_csv(csv_path)
        df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
        gestures = df["gesture"].values

        windows, labels = [], []
        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]
            window_label = gestures[start:end]
            # most frequent gesture
            majority = pd.Series(window_label).mode()[0] 
            windows.append(window)
            labels.append(gesture_to_idx.get(majority, 0))
        return np.stack(windows), np.array(labels)
