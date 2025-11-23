import torch
import torch.nn as nn

"""
Three heads models.
Uses seq2seq, next gesture classifier and time_to_gesture models.
"""
class MultiHeadHAR(nn.Module):
    def __init__(
        self,
        input_dim=6,
        hidden_dim=128,
        num_layers=2,
        num_classes=4,
        pred_steps=12
    ):
        super().__init__()

        # Shared encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Head 1: Seq2Seq Forecasting
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.seq2seq_fc = nn.Linear(hidden_dim, input_dim)
        self.pred_steps = pred_steps


        # Head 2: Next Gesture Classifier
        self.classifier_fc = nn.Linear(hidden_dim, num_classes)

        # Head 3: Time-to-Gesture Regressor
        self.regressor_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, 6]
        enc_out, (h, c) = self.encoder(x)

        last_hidden = enc_out[:, -1, :]  # [B, hidden]

        #  Classification 
        class_logits = self.classifier_fc(last_hidden)

        # Regression 
        time_regression = self.regressor_fc(last_hidden).squeeze(-1)

        # Seq2Seq decoder 
        B = x.size(0)
        decoder_input = enc_out[:, -1:, :]  # last time step

        preds = []
        hx, cx = h, c

        for _ in range(self.pred_steps):
            out, (hx, cx) = self.decoder_lstm(decoder_input, (hx, cx))
            step_pred = self.seq2seq_fc(out)  # [B,1,6]
            preds.append(step_pred)

            # Feed the prediction to next step
            decoder_input = out

        forecast = torch.cat(preds, dim=1)

        return forecast, class_logits, time_regression
