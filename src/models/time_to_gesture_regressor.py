# model_time_to_gesture.py
import torch
import torch.nn as nn
import math
from src.models.shared_layers import Encoder

class TimeToGestureRegressor(nn.Module):
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
            nn.Linear(64, 1),
            nn.Sigmoid()   
        )

    def forward(self, X_in):
        """
        X_in: [batch, T, 6]
        """
        encoder_outputs, (h, c) = self.encoder(X_in)

        # Final hidden state from encoder
        h_last = h[-1]  # shape: [batch, hidden_dim]

        # Predict time to next gesture
        y = self.time_head(h_last)

        return y.squeeze(-1)  # [batch]
