from src.time_to_gesture_dataset import TimeToGestureDataset
from src.time_to_gesture_model import TimeToGestureModel
import os
import zipfile
import glob
import torch
import numpy as np
import pandas as pd

ROOT = r"C:\Users\ammou\Documents\human-activity-recognition"
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extracted:", EXTRACT_DIR)
else:
    print("ZIP already extracted")

csv_files = glob.glob(os.path.join(EXTRACT_DIR, "*.csv"))
print("Found", len(csv_files), "CSV files")

gesture_to_idx = {
    "no_gesture": 0,
    "swipe_up": 1,
    "swipe_left": 2,
    "swipe_right": 3
}

all_imu = []
all_gest = []

for file in csv_files:
    print("Loading:", file)
    df = pd.read_csv(file)

    # Fill missing gestures
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values
    gest = df["gesture"].map(gesture_to_idx).values

    all_imu.append(imu)
    all_gest.append(gest)

imu_data = np.concatenate(all_imu, axis=0)
gesture_labels = np.concatenate(all_gest, axis=0)

print("IMU shape:", imu_data.shape)           # [T, 6]
print("Gesture labels:", gesture_labels.shape) # [T]

dataset = TimeToGestureDataset(
    imu_data=imu_data,
    gesture_labels=gesture_labels,
    window_size=24,     
    max_time=200
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("Dataset ready. Total samples:", len(dataset))
print("Example batch shapes:")
for batch in loader:
    x, y = batch
    print("x:", x.shape)  # [batch, window_size, 6]
    print("y:", y.shape)  # [batch]
    break



unique, counts = np.unique(gesture_labels, return_counts=True)

print("Class distribution:")
for cls, cnt in zip(unique, counts):
    print(f"Class {cls}: {cnt} samples")

import torch
from src.time_to_gesture_model import TimeToGestureModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TimeToGestureModel(
    input_dim=6,
    hidden_dim=128,
    num_layers=2
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(25):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)              # [batch]
        loss = criterion(pred, y)    # MSE loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
