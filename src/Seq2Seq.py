# seq2seq.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Encoder(nn.Module):
    def __init__(self, input_dim=6, d_model=128, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        self.lstm = nn.LSTM(
          input_size=d_model,
          hidden_size=hidden_dim,
          num_layers=num_layers,
          dropout=dropout if num_layers > 1 else 0.0,
          batch_first=True
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pe(x)
        outputs, (h, c) = self.lstm(x)  
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, output_dim=6, hidden_dim=128, num_layers=1, num_heads=16):
        super().__init__()
        self.lstm = nn.LSTM(
          input_size=output_dim,
          hidden_size=hidden_dim,
          num_layers=num_layers,
          batch_first=True
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, y_prev, hidden, encoder_outputs):
         out, hidden = self.lstm(y_prev, hidden)
         attn_out, attn_weights = self.attn(out, encoder_outputs, encoder_outputs)

         pred = self.fc(attn_out)  
         return pred, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing

    def forward(self, X_in, Y_out=None):
        B, T, C = X_in.shape
        device = X_in.device
        output, (h, c) = self.encoder(X_in)

        y_prev = X_in[:, -1:, :]  # last timestep
        preds = []
        all_attn = []
        for t in range(T):
            pred, (h, c), attn_w = self.decoder(y_prev, (h, c), output)
            preds.append(pred)
            all_attn.append(attn_w)
            if (Y_out is not None) and (torch.rand(1, device=device) < self.teacher_forcing):
                y_prev = Y_out[:, t:t+1, :]
            else:
                y_prev = pred.detach()

        return torch.cat(preds, dim=1), all_attn

def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X_in, Y_out in train_loader:
            X_in, Y_out = X_in.to(device), Y_out.to(device)
            optim.zero_grad()
            pred, _ = model(X_in, Y_out)
            loss = crit(pred, Y_out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_in, Y_out in val_loader:
                X_in, Y_out = X_in.to(device), Y_out.to(device)
                pred = model(X_in, Y_out=None)
                loss = crit(pred, Y_out)
                val_loss += loss.item()

        print(f"Epoch {ep:02d} | train MSE {train_loss/len(train_loader):.6f} "
              f"| val MSE {val_loss/len(val_loader):.6f}")
