import torch
from torch.utils.data import Dataset
import numpy as np

def find_gesture_onsets(labels):
    return [i for i in range(1, len(labels)) if labels[i] != 0 and labels[i-1] == 0]

def compute_next_gesture(labels):
    T = len(labels)
    onsets = find_gesture_onsets(labels)
    next_gest = np.zeros(T, dtype=np.int64)

    for i in range(T):
        future = [o for o in onsets if o >= i]
        if future:
            next_gest[i] = labels[future[0]]
        else:
            next_gest[i] = 0
    return next_gest

def compute_time_to_gesture(labels, max_time=200):
    T = len(labels)
    onsets = find_gesture_onsets(labels)
    ttg = np.full(T, max_time, dtype=np.int32)

    for i in range(T):
        future = [o for o in onsets if o >= i]
        if future:
            ttg[i] = future[0] - i

    return ttg / max_time


class MultiTaskIMUDataset(Dataset):
    def __init__(self, imu_data, gesture_labels, window_size=36, t_in=24, t_out=12, max_time=200):
        self.data = imu_data
        self.labels = gesture_labels
        self.window_size = window_size
        self.t_in = t_in
        self.t_out = t_out
        self.max_time = max_time

        # Precompute targets
        self.next_gesture = compute_next_gesture(self.labels)
        self.time_to_gesture = compute_time_to_gesture(self.labels, max_time)

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        T = len(self.data)
        for end in range(self.window_size, T - 1):
            start = end - self.window_size

            window = self.data[start:end]

            x_in = window[:self.t_in]
            y_out = window[self.t_in:self.t_in + self.t_out]

            cls = self.next_gesture[end]
            ttg = self.time_to_gesture[end]

            self.samples.append((x_in, y_out, cls, ttg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y_seq, cls, ttg = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
            torch.tensor(cls, dtype=torch.long),
            torch.tensor(ttg, dtype=torch.float32)
        )
