# seq2seq.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_dim=9, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x):
        _, (h, c) = self.rnn(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, out_dim=9, hidden=128, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, y_prev, hidden):
        out, hidden = self.rnn(y_prev, hidden)
        pred = self.fc(out)   # (B,1,C)
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing

    def forward(self, X_in, Y_out=None):
        B, T, C = X_in.shape
        device = X_in.device
        h, c = self.encoder(X_in)

        y_prev = X_in[:, -1:, :]  # last timestep
        preds = []

        for t in range(T):
            pred, (h, c) = self.decoder(y_prev, (h, c))
            preds.append(pred)

            if (Y_out is not None) and (torch.rand(1, device=device) < self.teacher_forcing):
                y_prev = Y_out[:, t:t+1, :]
            else:
                y_prev = pred.detach()

        return torch.cat(preds, dim=1)  # (B,T,C)


def train_loop(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X_in, Y_out in train_loader:
            X_in, Y_out = X_in.to(device), Y_out.to(device)
            optim.zero_grad()
            pred = model(X_in, Y_out)
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
