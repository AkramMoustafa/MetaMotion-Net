# train.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data_loader import IMUSeq2SeqDataset, split_train_val
from src.seq2seq import Encoder, Decoder, Seq2Seq, train_loop

if __name__ == "__main__":
    data = np.load("UCIHAR_train_seq.npz")
    X = data["X"]  # (7352,128,9)

    X_tr, X_va = split_train_val(X)
    mu = X_tr.mean(axis=(0,1), keepdims=True)
    sd = X_tr.std(axis=(0,1), keepdims=True)
    sd[sd < 1e-8] = 1.0

    T_IN = 127
    train_ds = IMUSeq2SeqDataset(X_tr, t_in=T_IN, mean_std=(torch.tensor(mu), torch.tensor(sd)))
    val_ds   = IMUSeq2SeqDataset(X_va, t_in=T_IN, mean_std=(train_ds.mu, train_ds.sd))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(in_dim=9, hidden=128)
    dec = Decoder(out_dim=9, hidden=128)
    model = Seq2Seq(enc, dec, teacher_forcing=0.5).to(device)

    train_loop(model, train_loader, val_loader, epochs=25, lr=1e-3, device=device)
