# src/models/time_to_gesture_regressor.py

import torch
import torch.nn as nn
from src.models.shared_layers import Encoder


class TimeToGestureRegressor(nn.Module):
    """
    Takes IMU window [batch, T, 6]
    Outputs single scalar per sample = normalized time until next gesture.
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

        self.attn = nn.Linear(hidden_dim, 1)

        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, X_in):
        """
        X_in: [batch, T, 6]
        """

        encoder_outputs, _ = self.encoder(X_in)

        weights = torch.softmax(self.attn(encoder_outputs), dim=1)

        h_attn = (weights * encoder_outputs).sum(dim=1)
        # shape: [batch, hidden_dim]

        y = self.time_head(h_attn)

        return y.squeeze(-1)
