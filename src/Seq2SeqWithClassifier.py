import torch
import torch.nn as nn
import math
from copy import deepcopy

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

    def forward(self, y_prev, hidden, encoder_outputs):
        h, c = hidden

        out, (h, c) = self.lstm(y_prev, (h, c))
        attn_out, attn_weights = self.attn(out, encoder_outputs, encoder_outputs)
        pred = self.fc(torch.tanh(attn_out))
        return pred, (h, c), attn_weights

class Seq2SeqWithClassifier(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing=0.3, pred_steps=12,
                 hidden_dim=128, num_classes=4, alpha=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pred_steps = pred_steps
        self.teacher_forcing = teacher_forcing
        self.out2hid = nn.Linear(6, hidden_dim)
        self.alpha = alpha

        self.gesture_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, X_in, Y_out=None):
        B, T, C = X_in.shape
        device = X_in.device

        encoder_outputs, (h, c) = self.encoder(X_in)
        gesture_logits = self.gesture_head(h[-1])  # [B, num_classes]

        preds = []
        for t in range(self.pred_steps):
            if t == 0:
                y_prev = torch.zeros(B, 1, 128, device=device)
            else:
                if (Y_out is not None) and (torch.rand(1).item() < self.teacher_forcing):
                    y_prev = self.out2hid(Y_out[:, t-1].unsqueeze(1))
                else:
                    y_prev = self.out2hid(preds[-1].detach())

            pred, (h, c), _ = self.decoder(y_prev, (h, c), encoder_outputs)
            preds.append(pred)

        return torch.cat(preds, dim=1), gesture_logits

def train_classifier_loop(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    alpha=0.5,
    device="cpu"
):
    model.to(device)

    # Losses
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=5
    )

    best_val = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    # EMA model
    ema_model = deepcopy(model)
    ema_decay = 0.99

    for ep in range(1, epochs + 1):

        # dynamic teacher forcing
        model.teacher_forcing = max(0.5 * (0.97 ** ep), 0.1)

        # EMA update
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_((1 - ema_decay) * param.data)

        model.train()
        train_loss = 0.0
        train_cls_correct = 0
        train_total = 0

        for X_in, Y_out, labels in train_loader:
            X_in  = X_in.to(device)
            Y_out = Y_out.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            pred_seq, gesture_logits = model(X_in, Y_out)

            loss_pred = mse_loss(pred_seq, Y_out)
            loss_cls  = ce_loss(gesture_logits, labels)

            loss = loss_pred + alpha * loss_cls
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_loss += loss.item()

            preds = gesture_logits.argmax(dim=1)
            train_cls_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_cls_correct / train_total

        ema_model.eval()
        val_loss = 0.0
        val_cls_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_in, Y_out, labels in val_loader:
                X_in  = X_in.to(device)
                Y_out = Y_out.to(device)
                labels = labels.to(device)

                pred_seq, gesture_logits = ema_model(X_in)

                loss_pred = mse_loss(pred_seq, Y_out)
                loss_cls  = ce_loss(gesture_logits, labels)

                loss = loss_pred + alpha * loss_cls
                val_loss += loss.item()

                preds = gesture_logits.argmax(dim=1)
                val_cls_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_cls_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {ep:02d} | TF={model.teacher_forcing:.3f} | "
              f"Train Loss={train_loss/len(train_loader):.6f} | "
              f"Train Acc={train_acc:.3f} | "
              f"Val Loss={val_loss/len(val_loader):.6f} | "
              f"Val Acc={val_acc:.3f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_classifier_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {best_val:.6f}")