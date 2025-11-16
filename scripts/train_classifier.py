import os
import glob
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from src.data_loader_classifer import IMUSeq2SeqClassifierDataset

ROOT_DIR = "C:/Users/ammou/Documents/human-activity-recognition"
DATA_DIR = os.path.join(ROOT_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "logs")

if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    print("Extracting ZIP...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print(f"Extracted to {EXTRACT_PATH}")
else:
    print("ZIP already extracted.")

GLOB_PATTERN = os.path.join(EXTRACT_PATH, "*.csv")
csv_files = glob.glob(GLOB_PATTERN)

print(f"Found {len(csv_files)} CSV files.")

gesture_to_idx = {
    "no_gesture": 0,
    "swipe_up": 1,
    "swipe_left": 2,
    "swipe_right": 3
}

WINDOW_SIZE = 36
STRIDE = 6

all_windows = []
all_labels = []

for file in csv_files:
    print("Processing:", file)

    w, l = IMUSeq2SeqClassifierDataset.build_window_labels(
        csv_path=file,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        gesture_to_idx=gesture_to_idx
    )

    all_windows.append(w)
    all_labels.append(l)

windows = np.concatenate(all_windows, axis=0)
labels  = np.concatenate(all_labels, axis=0)

print("FINAL windows:", windows.shape)
print("FINAL labels: ", labels.shape)


N, T, C = windows.shape
scaler = StandardScaler()
scaler.fit(windows.reshape(-1, C))
windows_norm = scaler.transform(windows.reshape(-1, C)).reshape(N, T, C).astype(np.float32)

n_val = int(0.2 * N)

val_windows = windows_norm[:n_val]
val_labels  = labels[:n_val]

train_windows = windows_norm[n_val:]
train_labels  = labels[n_val:]

print("Train:", train_windows.shape, train_labels.shape)
print("Val:  ", val_windows.shape, val_labels.shape)

train_ds = IMUSeq2SeqClassifierDataset(train_windows, train_labels)
val_ds   = IMUSeq2SeqClassifierDataset(val_windows, val_labels)

BATCH_SIZE = 128

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print("Dataloaders ready.")


from src.Seq2SeqWithClassifier import (
    Encoder, Decoder, Seq2SeqWithClassifier, train_classifier_loop
)
input_dim = 6
hidden_dim = 128
num_classes = len(gesture_to_idx)

encoder = Encoder(input_dim=input_dim, d_model=128, hidden_dim=128, num_layers=2)
decoder = Decoder(output_dim=6, hidden_dim=128, num_layers=2, num_heads=8)

model = Seq2SeqWithClassifier(
    encoder=encoder,
    decoder=decoder,
    teacher_forcing=0.3,
    pred_steps=12,
    hidden_dim=128,
    num_classes=num_classes,
    alpha=0.5   # weighting between regression & classification
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_classifier_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=40,
    lr=1e-3,
    alpha=0.5,
    device=DEVICE
)

