import os
import glob
import json
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib

from src.dataset.next_gesture_dataset import NextGestureDataset
from src.models.next_gesture_classifier import NextGestureClassifier
from src.train_utils import set_seed, get_device, save_model

CONFIG_PATH = "config/train_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

C = config["classifier"]
D = config["dataset"]
G = config["global"]

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")
SCALER_PATH = G["scaler_path"]

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extracted IMU log files.")
else:
    print("ZIP already extracted.")


print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

csv_files = glob.glob(os.path.join(EXTRACT_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")


all_imu = []
all_labels = []
gesture_to_idx = config["gesture_map"]
for file in csv_files:
    df = pd.read_csv(file)
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values
    gest = df["gesture"].map(gesture_to_idx).values

    all_imu.append(imu)
    all_labels.append(gest)

imu_data = np.concatenate(all_imu, axis=0)
gesture_labels = np.concatenate(all_labels, axis=0)

imu_data = scaler.transform(imu_data)

print("IMU:", imu_data.shape)
print("Labels:", gesture_labels.shape)

dataset = NextGestureDataset(
    imu_data=imu_data,
    gesture_labels=gesture_labels,
    window_size=D["window_size"]
)

loader = DataLoader(dataset, batch_size=C["batch_size"], shuffle=True)

set_seed(G["seed"])
device = get_device()

model = NextGestureClassifier(
    input_dim=6,
    hidden_dim=C["hidden_dim"],
    num_layers=C["num_layers"],
    num_classes=C["num_classes"]
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=C["learning_rate"])

for epoch in range(C["epochs"]):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{C['epochs']} - Loss={avg_loss:.4f}")

save_model(model, "next_gesture_classifier.pt")
print("Saved classifier model to next_gesture_classifier.pt")
