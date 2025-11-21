import torch
import torch.nn as nn

class NextGestureClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_classes=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last) 
        return logits
