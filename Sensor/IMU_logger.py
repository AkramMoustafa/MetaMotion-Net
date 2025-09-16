import sys
import time
import csv
import numpy as np
from datetime import datetime
from collections import deque
from scipy.signal import butter, filtfilt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from snaptic_sdk import PySnapticSDK

SAMPLE_RATE = 48 # Hz
WINDOW_SECONDS = 2.56  # window length
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)


class SnapticWorker(QThread):
    log_signal = pyqtSignal(str, str)

    def __init__(self, logfile=None):
        super().__init__()
        self.running = False
        self.snaptic = PySnapticSDK()
        self.logfile = logfile
        self.csv_file = None
        self.csv_writer = None

        # Rolling buffer for preprocessing
        self.buffer = deque(maxlen=WINDOW_SIZE)

    def preprocess_and_predict(self):
        """Preprocess buffer into model-ready window & predict."""
        if len(self.buffer) < WINDOW_SIZE:
            return None

        # create a window
        window = np.array(self.buffer)

        # remove mean from signals
        window = window - np.mean(window, axis=0, keepdims=True)

        # normalize output to be between [-1, 1]
        denom = np.maximum(np.max(np.abs(window), axis=0, keepdims=True), 1e-6)
        normed = window / denom

        # only pass values lower than 15hz
        b, a = butter(4, 15 / (SAMPLE_RATE / 2), btype="low")
        window = filtfilt(b, a, window, axis=0)

        # moving average smoothing
        kernel = np.ones(5) / 5
        normed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=normed
        )

        # TODO: replace with your ML model
        fake_pred = np.random.choice(["walking", "running", "idle"])
        return fake_pred

    def run(self):
        devices = self.snaptic.search_devices()
        if not devices:
            self.log_signal.emit("No devices found.", "error")
            return

        device = devices[0]
        if not self.snaptic.connect_device(device):
            self.log_signal.emit("Could not connect.", "error")
            return

        self.log_signal.emit(f"Connected to {device.Name}", "success")
        self.running = True

        # open CSV
        if self.logfile:
            self.csv_file = open(self.logfile, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "time", "packet_num", "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z"
            ])

        try:
            start = time.time()
            while self.running and time.time() - start < 60:  
                data = self.snaptic.get_imu_data()
                if data:
                    for pkt in data["packets"]:
                        acc_x, acc_y, acc_z = pkt["MainAccel"]["X"], pkt["MainAccel"]["Y"], pkt["MainAccel"]["Z"]
                        gyro_x, gyro_y, gyro_z = pkt["MainGyro"]["X"], pkt["MainGyro"]["Y"], pkt["MainGyro"]["Z"]

                        msg = (
                            f"Packet {pkt['PacketNum']} | "
                            f"Accel=({acc_x:.2f}, {acc_y:.2f}, {acc_z:.2f}) | "
                            f"Gyro=({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f})"
                        )
                        self.log_signal.emit(msg, "data")

                        if self.csv_writer:
                            self.csv_writer.writerow([
                                data["time"], pkt["PacketNum"],
                                acc_x, acc_y, acc_z,
                                gyro_x, gyro_y, gyro_z
                            ])

                        # push into buffer
                        self.buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

                        # run continuous prediction if buffer full
                        if len(self.buffer) == WINDOW_SIZE:
                            pred = self.preprocess_and_predict()
                            if pred:
                                self.log_signal.emit(f"Prediction: {pred}", "success")

                            # 50% overlap
                            for _ in range(WINDOW_SIZE // 2):
                                if self.buffer:
                                    self.buffer.popleft()

                time.sleep(0.02)
        finally:
            self.snaptic.disconnect_device()
            self.log_signal.emit("Disconnected.", "info")
            if self.csv_file:
                self.csv_file.close()

    def stop(self):
        self.running = False


class SnapticUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snaptic IMU Logger")
        self.setGeometry(200, 200, 700, 500)

        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: goldenrod; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 10))
        layout.addWidget(self.log_box)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Logging")
        self.start_btn.clicked.connect(self.start_logging)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_logging)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.worker = None

    def log(self, msg, msg_type="info"):
        colors = {"success": "green", "error": "red", "info": "blue", "data": "black"}
        color = colors.get(msg_type, "black")
        self.log_box.setTextColor(QColor(color))
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

        self.status_label.setText("Status: " + msg)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def start_logging(self):
        if not self.worker or not self.worker.isRunning():
            logfile = f"snaptic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.worker = SnapticWorker(logfile=logfile)
            self.worker.log_signal.connect(self.log)
            self.worker.start()
            self.log(f"Logging to {logfile}", "info")

    def stop_logging(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested...", "info")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapticUI()
    window.show()
    sys.exit(app.exec_())
