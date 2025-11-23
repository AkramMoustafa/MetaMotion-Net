# seq2seq.py
import torch
import torch.nn as nn
import math
from copy import deepcopy
from src.models.shared_layers import Encoder

class Decoder(nn.Module):
    def __init__(self, output_dim=6, hidden_dim=128, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, y_prev, hidden, encoder_outputs):
        h, c = hidden

        # Match LSTM layer count
        if h.size(0) != self.lstm.num_layers:
            h = h[-self.lstm.num_layers:]
            c = c[-self.lstm.num_layers:]

        # Decode one timestep
        out, hidden = self.lstm(y_prev, (h, c))
        attn_out, attn_weights = self.attn(out, encoder_outputs, encoder_outputs)
        # out = out + self.dropout(attn_out)
        
        pred = self.fc(torch.tanh(attn_out))
        return pred, hidden, attn_weights


class Seq2SeqForecaster(nn.Module):
    
    
    def __init__(self, encoder, decoder, teacher_forcing=0.3, pred_steps=12):
        super().__init__()
        self.pred_steps = pred_steps
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.out2hid = nn.Linear(6,128)

    def forward(self, X_in, Y_out=None):
        B, T, C = X_in.shape
        device = X_in.device

        # Encode input
        output, (h, c) = self.encoder(X_in)

        preds = []
        all_attn = []
        for t in range(self.pred_steps):
        
            if t == 0:
                # first decoder input = zero vector
                y_prev = torch.zeros(B, 1, 128, device=device)
            else:
                if (Y_out is not None) and (torch.rand(1).item() < self.teacher_forcing):
                    # teacher forcing: use ground truth at step t-1
                    y_prev = self.out2hid(Y_out[:, t-1].unsqueeze(1))
                    
                else:
                    # use model's own last prediction
                    y_prev = self.out2hid(preds[-1].detach())
          
            # Run one decoder step
            pred, (h, c), attn_w = self.decoder(y_prev, (h, c), output)
            preds.append(pred)
            all_attn.append(attn_w)

        return torch.cat(preds, dim=1), all_attn

def train_loop(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    patience_counter = 0       
    early_stop_patience = 10  

    ema_decay = 0.99
    ema_model = deepcopy(model)
    for ep in range(1, epochs + 1):
        # Dynamic teacher forcing
        model.teacher_forcing = max(0.5 * (0.97 ** ep), 0.1)
        with torch.no_grad():
          for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_((1 - ema_decay) * param.data)
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
                pred, _ = ema_model(X_in)
                loss = crit(pred, Y_out)
                val_loss += loss.item()

        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step(val_loss)

        # Check if val loss improved
        if val_loss < best_val - 1e-5:  
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_seq2seq_model.pt")
        else:
            patience_counter += 1

        print(f"Epoch {ep:02d} | TF={model.teacher_forcing:.3f} | "
              f"train MSE {train_loss:.6f} | val MSE {val_loss:.6f} | "
              f"LR {optim.param_groups[0]['lr']:.6e}")

        # Early stopping condition
        if patience_counter >= early_stop_patience:
            print(f"?? Early stopping triggered after {ep} epochs (no improvement for {early_stop_patience}).")
            break

    print(f"? Training complete. Best val MSE: {best_val:.6f}")

