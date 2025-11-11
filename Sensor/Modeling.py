# realtime_predict.py
import sys
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import torch
import joblib
from src.Seq2Seq import Encoder, Decoder, Seq2Seq
from Sensor.IMU_logger import SnapticWorker  

SAMPLE_RATE = 48           
T_IN        = 24           
T_OUT       = 12
WINDOW      = T_IN + T_OUT    
STRIDE      = 6             
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH  = "./Trained_Models/best_seq2seq_model.pt"
SCALER_PATH = "./Trained_Models/scaler.pkl"

CHANNEL_NAMES = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]


class LivePredictUI(QWidget):
    """
    Subscribes to SnapticWorker.sample_signal, keeps a rolling buffer of 6D IMU,
    runs Seq2Seq to predict the next T_OUT steps, and overlays lines:
      - solid = actual last N points of a chosen channel
      - dashed = predicted next T_OUT for that same channel (advanced on the x axis)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time IMU — Prediction vs Actual")
        self.setGeometry(200, 200, 1000, 620)

        # Layout 
        root = QVBoxLayout()

        top = QHBoxLayout()
        self.status = QLabel("Status: Idle")
        self.status.setStyleSheet("font-weight:600;")
        top.addWidget(self.status)

        top.addStretch(1)

        top.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(CHANNEL_NAMES)
        self.channel_combo.setCurrentIndex(0)  # default acc_x
        top.addWidget(self.channel_combo)

        self.start_btn = QPushButton("Start Stream")
        self.start_btn.clicked.connect(self.start_stream)
        top.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_stream)
        top.addWidget(self.stop_btn)

        root.addLayout(top)

        #  Plot 
        self.plot = pg.PlotWidget(background="w")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time (samples)")
        self.plot.setLabel("left", "Sensor value")
        self.curve_actual = self.plot.plot(pen=pg.mkPen((30, 100, 200), width=2), name="Actual")
        self.curve_pred   = self.plot.plot(pen=pg.mkPen((200, 50, 50), width=2, style=Qt.DashLine), name="Predicted")
        root.addWidget(self.plot)

        self.setLayout(root)

        # State 
        self.buffer = deque(maxlen=WINDOW * 6)  # store raw samples flattened: len = maxlen, each push 6 values
        self.actual_vals = deque(maxlen=WINDOW * 8)  # for plotting longer actual history of the chosen channel
        self.last_pred = None  # (T_OUT, 6) numpy

        # used to trigger prediction every STRIDE samples
        self._sample_counter = 0

        #  Load model + scaler 
        self.scaler = joblib.load(SCALER_PATH)

        enc = Encoder(input_dim=6, d_model=128, hidden_dim=128, num_layers=2)
        dec = Decoder(output_dim=6, hidden_dim=128, num_heads=8, num_layers=2)
        self.model = Seq2Seq(enc, dec, teacher_forcing=0.0, pred_steps=T_OUT).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        # Worker 
        self.worker = None

        #  UI refresh timer 
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # ~20 Hz UI update
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start()

    # Stream Control 
    def start_stream(self):
        if self.worker and self.worker.isRunning():
            self.status.setText("Status: Already running")
            return
        self.worker = SnapticWorker(logfile=None)  # no CSV needed for live UI, keep your device config as-is
        self.worker.log_signal.connect(self.on_log)
        self.worker.sample_signal.connect(self.on_sample)  # receive np.array([6])
        self.worker.start()
        self.status.setText("Status: Streaming...")

    def stop_stream(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status.setText("Status: Stop requested...")

    # Signals
    def on_log(self, msg, msg_type="info"):
        # surface minimal status; your logger already prints elsewhere
        if msg_type == "error":
            self.status.setText(f"Status: ERROR — {msg}")
        elif msg_type == "success":
            self.status.setText(f"Status: {msg}")
        # otherwise keep the current status text

    def on_sample(self, vec6):
        """
        vec6: shape (6,), order [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        """
        # append to flattened buffer
        self.buffer.extend(vec6.tolist())

        # update actual history for the currently selected channel
        ch = self.channel_combo.currentIndex()
        self.actual_vals.append(vec6[ch])

        # run model every STRIDE samples if we have enough for a window
        self._sample_counter = (self._sample_counter + 1) % STRIDE
        if self._have_full_window() and self._sample_counter == 0:
            self._run_model_on_tail()

    # Inference 
    def _have_full_window(self):
       
        return len(self.buffer) >= WINDOW * 6

    def _run_model_on_tail(self):
        """
        Take the last WINDOW samples from buffer (shape WINDOW x 6),
        normalize using the saved scaler, feed first T_IN to the model,
        get next T_OUT prediction.
        """
        arr = np.array(self.buffer, dtype=np.float32).reshape(-1, 6)  # (N, 6)
        tail = arr[-WINDOW:] 

        
        tail_norm = self.scaler.transform(tail) 
    
        x_in = torch.from_numpy(tail_norm[:T_IN]).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():
            pred_norm, _ = self.model(x_in)  # (1, T_OUT, 6)

        pred_norm = pred_norm.squeeze(0).cpu().numpy()  # (T_OUT, 6)

        pred_real = self.scaler.inverse_transform(pred_norm)  # (T_OUT, 6)

        self.last_pred = pred_real  # store for plotting

    def refresh_plot(self):
        # update actual curve
        if len(self.actual_vals) > 1:
            y_actual = np.array(self.actual_vals, dtype=float)
            x_actual = np.arange(len(y_actual))
            self.curve_actual.setData(x=x_actual, y=y_actual)

        if self.last_pred is not None and len(self.actual_vals) > 0:
            n = len(self.actual_vals)
            y_pred = self.last_pred[:, self.channel_combo.currentIndex()]
            x_pred = np.arange(n, n + len(y_pred))
            self.curve_pred.setData(x=x_pred, y=y_pred)

    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = LivePredictUI()
    ui.show()
    sys.exit(app.exec_())
