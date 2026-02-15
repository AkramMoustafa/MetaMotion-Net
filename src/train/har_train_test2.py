import os
import zipfile
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import joblib

from src.config_loader import load_config
from src.dataset.har_dataset import MultiTaskIMUDataset
from src.models.multihead_har_model import MultiHeadHAR
from src.train_utils import set_seed, get_device

cfg = load_config()
G = cfg["global"]
D = cfg["dataset"]
S = cfg["seq2seq"]
C = cfg["classifier"]
T = cfg["time_to_gesture"]
gesture_map = cfg["gesture_map"]

set_seed(G["seed"])
device = get_device()

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "logs")
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

    # Fill gesture labels
    df["gesture"] = df["gesture"].ffill().fillna("no_gesture")

    imu = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values
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

n_total = len(dataset)
n_test = int(0.15 * n_total)
n_val = int(0.15 * n_total)
n_train = n_total - n_test - n_val

train_ds, val_ds, test_ds = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(G["seed"])
)

print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

train_loader = DataLoader(train_ds, batch_size=S["batch_size"], shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=S["batch_size"], shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=S["batch_size"], shuffle=False)

model = MultiHeadHAR(
    input_dim=G["input_dim"],
    hidden_dim=S["hidden_dim"],
    num_layers=S["num_layers"],
    num_classes=C["num_classes"],
    pred_steps=S["t_out"]
).to(device)

mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=S["learning_rate"],
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=True
)

epochs = S["epochs"]
best_val_loss = float("inf")

# Loss weights (VERY important for multitask)
LOSS_W_SEQ = 1.0
LOSS_W_CLS = 2.0
LOSS_W_TTG = 0.5

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0

    for x, y_seq, y_cls, y_ttg in train_loader:
        x = x.to(device)
        y_seq = y_seq.to(device)
        y_cls = y_cls.to(device)
        y_ttg = y_ttg.to(device)

        optimizer.zero_grad()

        pred_seq, pred_cls, pred_ttg = model(x)

        loss_seq = mse(pred_seq, y_seq)
        loss_cls = ce(pred_cls, y_cls)
        loss_ttg = mse(pred_ttg, y_ttg)

        loss = (
            LOSS_W_SEQ * loss_seq +
            LOSS_W_CLS * loss_cls +
            LOSS_W_TTG * loss_ttg
        )

        loss.backward()

        # ? Gradient clipping (critical)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y_seq, y_cls, y_ttg in val_loader:
            x = x.to(device)
            y_seq = y_seq.to(device)
            y_cls = y_cls.to(device)
            y_ttg = y_ttg.to(device)

            pred_seq, pred_cls, pred_ttg = model(x)

            loss_seq = mse(pred_seq, y_seq)
            loss_cls = ce(pred_cls, y_cls)
            loss_ttg = mse(pred_ttg, y_ttg)

            loss = (
                LOSS_W_SEQ * loss_seq +
                LOSS_W_CLS * loss_cls +
                LOSS_W_TTG * loss_ttg
            )

            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)

    print(
        f"Epoch {epoch}/{epochs} | "
        f"Train Loss={avg_train:.4f} | "
        f"Val Loss={avg_val:.4f}"
    )

    # Step scheduler
    scheduler.step(avg_val)

    # Save best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "multitask_best.pt")
        print("? Saved best model")


torch.save(model.state_dict(), "multitask_model_test3.pt")
print("Saved model multitask_model.pt")

model.eval()
test_loss = 0.0

with torch.no_grad():
    for x, y_seq, y_cls, y_ttg in test_loader:
        x = x.to(device)
        y_seq = y_seq.to(device)
        y_cls = y_cls.to(device)
        y_ttg = y_ttg.to(device)

        pred_seq, pred_cls, pred_ttg = model(x)

        loss_seq = mse(pred_seq, y_seq)
        loss_cls = ce(pred_cls, y_cls)
        loss_ttg = mse(pred_ttg, y_ttg)

        test_loss += (
            LOSS_W_SEQ * loss_seq +
            LOSS_W_CLS * loss_cls +
            LOSS_W_TTG * loss_ttg
        ).item()

avg_test = test_loss / len(test_loader)
print(f"\nFinal TEST Loss = {avg_test:.4f}")