import torch
import pandas as pd
import numpy as np
from src.data_loader import IMUSeq2SeqDataset

class IMUSeq2SeqClassifierDataset(IMUSeq2SeqDataset):
    """
    Same as IMUSeq2SeqDataset but returns gesture labels too.
    Used for predicting the gesture happening within each window.
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
        """
        Build windows and assign each window a single gesture label
        """
        df = pd.read_csv(csv_path)
        df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
        gestures = df["gesture"].values

        windows, labels = [], []

        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]
            window_label = gestures[start:end]

            # If all frames are 'no_gesture', keep as 'no_gesture'
            if np.all(window_label == "no_gesture"):
                label = "no_gesture"
            else:
                # Filter out 'no_gesture' and pick the most frequent active gesture
                non_no = [g for g in window_label if g != "no_gesture"]
                label = pd.Series(non_no).mode()[0] if len(non_no) > 0 else "no_gesture"

            windows.append(window)
            labels.append(gesture_to_idx.get(label, 0))

        return np.stack(windows), np.array(labels)

class IMUSeq2SeqDualLabelDataset(IMUSeq2SeqDataset):
    """
    Extends the base Seq2Seq dataset to include both current and next gesture prediction.
    """

    def __init__(self, windows, window_labels, next_labels, t_in=24, t_out=12, **kwargs):
        super().__init__(windows, t_in=t_in, t_out=t_out, **kwargs)

        assert len(window_labels) == len(windows)
        assert len(next_labels) == len(windows)
        self.window_labels = torch.tensor(window_labels, dtype=torch.long)
        self.next_labels   = torch.tensor(next_labels, dtype=torch.long)

    def __getitem__(self, idx):
        x_in, y_out = super().__getitem__(idx)
        label_now   = self.window_labels[idx]
        label_next  = self.next_labels[idx]
        return x_in, y_out, label_now, label_next

    @staticmethod
    def build_dual_labels(all_labels):
        """
        Create next-step labels by shifting window labels by one
        """
        next_labels = np.roll(all_labels, -1)
        next_labels[-1] = all_labels[-1]  
        return next_labels
