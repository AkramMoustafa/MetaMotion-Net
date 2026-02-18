import os
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib

from src.config_loader import load_config
from src.dataset.seq2seq_dataset import IMUSeq2SeqDataset
from src.models.seq2seq_forecaster import Encoder, Decoder, Seq2SeqForecaster, train_loop
from src.train_utils import set_seed, get_device, save_model

cfg = load_config()

global_cfg = cfg["global"]
seq_cfg    = cfg["seq2seq"]

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "logs")

SCALER_PATH = os.path.join(ROOT, seq_cfg["scaler_name"])
MODEL_SAVE_PATH = os.path.join(ROOT, seq_cfg["model_name"])

if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Extracted log files.")
else:
    print("ZIP already extracted.")

DEVICE = get_device()
set_seed(global_cfg["seed"])
print("Using device:", DEVICE)

GLOB_PATTERN = os.path.join(EXTRACT_PATH, "*.csv")

windows = IMUSeq2SeqDataset.combine_csv_files(
    pattern=GLOB_PATTERN,
    window_size=seq_cfg["window_size"],
    stride=seq_cfg["stride"]
)

print("Raw windows:", windows.shape)  # (N, 36, 6)

N, T, C = windows.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(windows.reshape(-1, C))

windows_norm = scaler.transform(windows.reshape(-1, C)).reshape(N, T, C)

n_total = len(windows_norm)
n_val   = int(seq_cfg["val_split"] * n_total)

train_windows = windows_norm[n_val:]
val_windows   = windows_norm[:n_val]

train_ds = IMUSeq2SeqDataset(train_windows, t_in=seq_cfg["t_in"], t_out=seq_cfg["t_out"])
val_ds   = IMUSeq2SeqDataset(val_windows,   t_in=seq_cfg["t_in"], t_out=seq_cfg["t_out"])

train_loader = DataLoader(
    train_ds,
    batch_size=seq_cfg["batch_size"],
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=seq_cfg["batch_size"],
    shuffle=False,
    drop_last=False
)

encoder = Encoder(
    input_dim=global_cfg["input_dim"],
    d_model=seq_cfg["d_model"],
    hidden_dim=seq_cfg["hidden_dim"],
    num_layers=seq_cfg["num_layers"]
)

decoder = Decoder(
    output_dim=global_cfg["input_dim"],
    hidden_dim=seq_cfg["hidden_dim"],
    num_heads=seq_cfg["num_heads"]
)

model = Seq2SeqForecaster(
    encoder,
    decoder,
    teacher_forcing=seq_cfg["teacher_forcing"],
    pred_steps=seq_cfg["t_out"]
).to(DEVICE)

train_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=seq_cfg["epochs"],
    lr=seq_cfg["learning_rate"],
    device=DEVICE
)

joblib.dump(scaler, SCALER_PATH)
save_model(model, MODEL_SAVE_PATH)

print(f"Training complete.")
print(f"Model saved → {MODEL_SAVE_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
