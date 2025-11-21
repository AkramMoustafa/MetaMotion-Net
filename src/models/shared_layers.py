import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 

        self.register_buffer("pe", pe.unsqueeze(0)) 

    def forward(self, x):
        T = x.size(1) 
        return x + self.pe[:, :T, :]
class Encoder(nn.Module):

    def __init__(self, input_dim=6, d_model=128, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        x = self.input_proj(x)       # [B, T, d_model]
        x = self.pos_enc(x)          # add positional encoding
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)
