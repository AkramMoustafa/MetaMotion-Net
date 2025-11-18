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
    def __init__(self, input_dim=6, d_model=128, hidden_dim=128, num_layers=2, dropout=0.3):
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

class TimeToGestureModel(nn.Module):
    """
    Takes IMU window [batch, T, 6]
    Outputs single scalar per sample = time until next gesture.
    """
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            d_model=128,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.3
        )

        # Regression head → predicts time until next gesture (frames)
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X_in):
        """
        X_in: [batch, T, 6]
        """
        encoder_outputs, (h, c) = self.encoder(X_in)

        h_last = h[-1]  # shape: [batch, hidden_dim]

        # Predict time to next gesture
        y = self.time_head(h_last)

        return y.squeeze(-1)  # [batch]
