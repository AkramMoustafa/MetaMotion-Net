import torch
import torch.nn as nn
import math

class Seq2SeqWithClassifier(nn.Module):
    def __init__(self, encoder, decoder, num_classes, teacher_forcing=0.5, alpha=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.alpha = alpha  
        self.classifier_fc = nn.Linear(encoder.lstm.hidden_size, num_classes)

    def forward(self, X_in, Y_out=None):
        B, T, C = X_in.shape
        device = X_in.device

        encoder_outputs, (h, c) = self.encoder(X_in)

        # Take the last hidden state
        h_last = h[-1]
        # put it in neural network for classification
        logits = self.classifier_fc(h_last)

        y_prev = X_in[:, -1:, :]
        preds = []
        for t in range(T-1):
            pred, (h, c), _ = self.decoder(y_prev, (h, c), encoder_outputs)
            preds.append(pred)
            if (Y_out is not None) and (torch.rand(1, device=device) < self.teacher_forcing):
                y_prev = Y_out[:, t:t+1, :]
            else:
                y_prev = pred.detach()

        forecast = torch.cat(preds, dim=1)
        return forecast, logits

def train_joint(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X_in, Y_out, y_label in train_loader:
            X_in, Y_out, y_label = X_in.to(device), Y_out.to(device), y_label.to(device)

            optim.zero_grad()
            forecast, logits = model(X_in, Y_out)

            loss_forecast = mse_loss(forecast, Y_out)
            loss_classify = ce_loss(logits, y_label) 
            loss = (1 - model.alpha) * loss_forecast + model.alpha * loss_classify

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
            
        print(f"Epoch {ep:02d} | train joint loss {loss_forecast/len(train_loader):.4f} | Loss Classify {loss_classify}")
