import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from multitask_src.data_loader import IMUSeq2SeqClassifierDataset
from multitask_src.model_seq2seq_classifier import Encoder, Decoder, Seq2SeqWithClassifier, train_loop

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WINDOW_SIZE = 36
STRIDE = 6
T_IN = 24
T_OUT = 12
NUM_CLASSES = 4

DATA_DIR = "data/logs"
gesture_to_idx = {
    "no_gesture": 0,
    "swipe_left": 1,
    "swipe_right": 2,
    "swipe_up": 3,
}

print("Loading IMU data windows...")

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
all_windows, all_labels = [], []

for f in csv_files:
    path = os.path.join(DATA_DIR, f)
    try:
        windows, labels = IMUSeq2SeqClassifierDataset.build_window_labels(
            path, WINDOW_SIZE, STRIDE, gesture_to_idx
        )
        all_windows.append(windows)
        all_labels.append(labels)
    except Exception as e:
        print(f"⚠️ Skipping {f}: {e}")

all_windows = np.concatenate(all_windows, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Loaded {len(all_windows)} windows")
print(f"Label distribution: {np.unique(all_labels, return_counts=True)}")

# Split train/val
n = len(all_windows)
split = int(0.8 * n)
train_w, val_w = all_windows[:split], all_windows[split:]
train_l, val_l = all_labels[:split], all_labels[split:]

train_ds = IMUSeq2SeqClassifierDataset(train_w, train_l, t_in=T_IN, t_out=T_OUT)
val_ds = IMUSeq2SeqClassifierDataset(val_w, val_l, t_in=T_IN, t_out=T_OUT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

encoder = Encoder()
decoder = Decoder()
model = Seq2SeqWithClassifier(encoder, decoder, num_classes=NUM_CLASSES).to(device)

print("Model initialized")

train_loop(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

print(" Training complete. Best model saved as 'best_seq2seq_classifier.pt'")
