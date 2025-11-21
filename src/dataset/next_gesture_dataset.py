import torch
from torch.utils.data import Dataset
import numpy as np


def find_gesture_onsets(labels):
    onsets = []
    for i in range(1, len(labels)):
        if labels[i] != 0 and labels[i - 1] == 0:
            onsets.append(i)
    return onsets


def compute_next_gesture(labels):
    T = len(labels)
    onsets = find_gesture_onsets(labels)
    next_gesture = np.zeros(T, dtype=np.int64)

    for i in range(T):
        future = [o for o in onsets if o >= i]
        if future:
            onset = future[0]
            next_gesture[i] = labels[onset]
        else:
            next_gesture[i] = 0

    return next_gesture


class NextGestureDataset(Dataset):
    def __init__(self, imu_data, gesture_labels, window_size=24):
        self.data = imu_data
        self.labels = gesture_labels
        self.window_size = window_size

        # Compute next gesture class
        self.next_gesture = compute_next_gesture(self.labels)

        # Build samples
        self.samples = []
        self.build_samples()

    def build_samples(self):
        T = len(self.data)
        for end in range(self.window_size, T - 1):
            start = end - self.window_size
            x = self.data[start:end]
            y = self.next_gesture[end]  # class label
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)  # CE needs long
        return x, y
