#Model Training File
# Takes 36 samples and creates windows with overlap of 6 which is 50 
# Prediction is continuous using Multihead attention from the research paper attention is all you need.
# Positional Encoding and Deciding are done under src.Seq2Seq code which includes the pipeline for the model  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib 
import zipfile
import os
from src.data_loader import IMUSeq2SeqDataset
from src.Seq2Seq import Encoder, Decoder, Seq2Seq, train_loop

#Test 1
GLOB_PATTERN = "*.csv"
WINDOW_SIZE  = 36
STRIDE       = 6
T_IN         = 24
T_OUT        = 12
BATCH_SIZE   = 128
NUM_WORKERS  = 0
EPOCHS       = 300
LR           = 1e-3

DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "snaptic_logs.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "logs")

# Extract only if not already extracted
if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print(f"Extracted logs to {EXTRACT_PATH}")

# Now point glob to extracted files
GLOB_PATTERN = os.path.join(EXTRACT_PATH, "*.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

windows = IMUSeq2SeqDataset.combine_csv_files(GLOB_PATTERN, window_size=WINDOW_SIZE, stride=STRIDE)
print("Raw windows:", windows.shape)  # (N, T, C)

N, T, C = windows.shape
scaler = StandardScaler()
scaler.fit(windows.reshape(-1, C))
windows_norm = scaler.transform(windows.reshape(-1, C)).reshape(N, T, C)

n_total = len(windows_norm)
n_val   = int(0.2 * n_total)
train_windows = windows_norm[n_val:]
val_windows   = windows_norm[:n_val]

train_ds = IMUSeq2SeqDataset(train_windows, t_in=T_IN, t_out=T_OUT, normalize=False, mean_removal=False)
val_ds   = IMUSeq2SeqDataset(val_windows,   t_in=T_IN, t_out=T_OUT, normalize=False, mean_removal=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

encoder = Encoder(input_dim=6, d_model=128, hidden_dim=128, num_layers = 2)
decoder = Decoder(output_dim=6, hidden_dim=128, num_heads=8)
model   = Seq2Seq(encoder, decoder, teacher_forcing=0.3, pred_steps=T_OUT).to(DEVICE)

train_loop(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

torch.save(model.state_dict(), "seq2seq_model.pt")
joblib.dump(scaler, "scaler.pkl")

print("? Training complete. Model saved to seq2seq_model.pt and scaler.pkl")