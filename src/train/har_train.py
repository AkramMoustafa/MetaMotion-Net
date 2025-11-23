import os
import zipfile
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

from src.config_loader import load_config
from src.dataset.har_dataset import MultiTaskIMUDataset
from src.models.seq2seq_forecaster import Encoder, Decoder, Seq2SeqForecaster
from src.models.multihead_har_model import MultiHeadHAR
from src.train_utils import set_seed, get_device

cfg = load_config()
G = cfg["global"]
D = cfg["dataset"]
S = cfg["seq2seq"]
C = cfg["classifier"]
T = cfg["time_to_gesture"]
gesture_map = cfg["gesture_map"]

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")

# scaler path
SCALER_PATH = os.path.join(ROOT, G["scaler_path"])

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extracted IMU logs.")
else:
    print("IMU logs already extracted.")

csv_files = glob.glob(os.path.join(EXTRACT_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV Files.")

all_imu, all_labels = [], []

for f in csv_files:
    df = pd.read_csv(f)
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].values
    gest = df["gesture"].map(gesture_map).values

    all_imu.append(imu)
    all_labels.append(gest)

imu_data = np.concatenate(all_imu, axis=0)
gesture_labels = np.concatenate(all_labels, axis=0)

print("IMU shape:", imu_data.shape)
print("Gesture labels:", gesture_labels.shape)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)
imu_data = scaler.transform(imu_data)

dataset = MultiTaskIMUDataset(
    imu_data=imu_data,
    gesture_labels=gesture_labels,
    window_size=S["window_size"],
    t_in=S["t_in"],
    t_out=S["t_out"],
    max_time=D["max_time_to_gesture"]
)

loader = DataLoader(dataset, batch_size=S["batch_size"], shuffle=True)

set_seed(G["seed"])
device = get_device()
encoder = Encoder(
    input_dim=G["input_dim"],
    d_model=S["d_model"],
    hidden_dim=S["hidden_dim"],
    num_layers=S["num_layers"]
)

decoder = Decoder(
    output_dim=G["input_dim"],
    hidden_dim=S["hidden_dim"],
    num_heads=S["num_heads"]
)


model = MultiHeadHAR(
    input_dim=G["input_dim"],
    hidden_dim=S["hidden_dim"],
    num_layers=S["num_layers"],
    num_classes=C["num_classes"],
    pred_steps=S["t_out"]
).to(device)


# losses
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=S["learning_rate"])

epochs = S["epochs"]

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for x, y_seq, y_cls, y_ttg in loader:
        x = x.to(device)
        y_seq = y_seq.to(device)
        y_cls = y_cls.to(device)
        y_ttg = y_ttg.to(device)

        optimizer.zero_grad()

        pred_seq, pred_cls, pred_ttg = model(x)

        loss_seq = mse(pred_seq, y_seq)
        loss_cls = ce(pred_cls, y_cls)
        loss_ttg = mse(pred_ttg, y_ttg)

        loss = loss_seq + loss_cls + loss_ttg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f"Epoch {epoch}/{epochs} | Loss={avg:.4f}")

torch.save(model.state_dict(), "multitask_model.pt")
print("Saved model multitask_model.pt")
