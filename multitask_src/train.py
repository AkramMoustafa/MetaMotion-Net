
import numpy as np
from src.data_loader import IMUSeq2SeqDataset, split_train_val
from src.Seq2Seq import Encoder, Decoder, Seq2Seq, train_loop
from  src.Seq2SeqWithClassifier import Seq2SeqWithClassifier, train_joint
import torch
from torch.utils.data import DataLoader
t_in = 24  

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build encoder/decoder
encoder = Encoder(input_dim=6, d_model=128, hidden_dim=128)
decoder = Decoder(output_dim=6, hidden_dim=128, num_heads=8)
# model = Seq2Seq(encoder, decoder, teacher_forcing=0.5)

X_all, Y_all, classes = IMUSeq2SeqDataset.combine_csv_files("*.csv")

# Create dataset
dataset = IMUSeq2SeqDataset(X_all, Y_all, t_in)

# Inspect one sample
x_in, y_out, y_label = dataset[0]
X_train, Y_train, X_val, Y_val = split_train_val(X_all, Y_all, val_ratio=0.2)

train_ds = IMUSeq2SeqDataset(X_train, Y_train, t_in)
val_ds   = IMUSeq2SeqDataset(X_val,   Y_val, t_in)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, drop_last=False)

model = Seq2SeqWithClassifier(
    encoder=encoder,
    decoder=decoder,
    num_classes=len(classes),
    teacher_forcing=0.5,
    alpha=0.5
).to(device)

train_joint(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    device=device
)

