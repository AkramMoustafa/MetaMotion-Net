import os
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

from train.config_loader import load_config
from dataset.time_to_gesture_dataset import TimeToGestureDataset
from models.time_to_gesture_regressor import TimeToGestureRegressor
from train.train_utils import set_seed, get_device, save_model

# Load config.json
cfg = load_config()
global_cfg = cfg["global"]
task_cfg = cfg["time_to_gesture"]

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")

SCALER_PATH = os.path.join(ROOT, global_cfg["scaler_path"])
MODEL_SAVE_PATH = os.path.join(ROOT, global_cfg["save_dir"], task_cfg["model_name"])

# Ensure save directory exists
os.makedirs(os.path.join(ROOT, global_cfg["save_dir"]), exist_ok=True)

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extracted log files.")
else:
    print("ZIP already extracted.")

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

csv_files = glob.glob(os.path.join(EXTRACT_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")

gesture_to_idx = cfg["global"]["gesture_map"]

all_imu, all_labels = [], []

for file in csv_files:
    df = pd.read_csv(file)
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
    gest = df["gesture"].map(gesture_to_idx).values

    all_imu.append(imu)
    all_labels.append(gest)

imu_data = np.concatenate(all_imu, axis=0)
gesture_labels = np.concatenate(all_labels, axis=0)

# Apply scaling
imu_data = scaler.transform(imu_data)

print("IMU:", imu_data.shape)
print("Labels:", gesture_labels.shape)

dataset = TimeToGestureDataset(
    imu_data=imu_data,
    gesture_labels=gesture_labels,
    window_size=task_cfg["window_size"],
    max_time=task_cfg["max_time"]
)

loader = DataLoader(
    dataset,
    batch_size=task_cfg["batch_size"],
    shuffle=True
)

set_seed(global_cfg["seed"])
device = get_device()

model = TimeToGestureRegressor(
    input_dim=global_cfg["input_dim"],
    hidden_dim=task_cfg["hidden_dim"],
    num_layers=task_cfg["num_layers"]
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=task_cfg["learning_rate"])

epochs = task_cfg["epochs"]

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f}")

save_model(model, MODEL_SAVE_PATH)
print(f"Model saved → {MODEL_SAVE_PATH}")
