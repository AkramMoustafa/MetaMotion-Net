import sys
import time
import csv
import numpy as np
from datetime import datetime
from collections import deque
from scipy.signal import butter, filtfilt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont
from snaptic_sdk import PySnapticSDK

SAMPLE_RATE = 48  # Hz
WINDOW_SECONDS = 2.65  # window length
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)

AVAILABLE_GESTURES = [
    "no_gesture",
    "swipe_up",
    "swipe_left",
    "swipe_right",
]

class SnapticWorker(QThread):
    # One-shot gesture from UI
    gesture_once = pyqtSignal(str)
    # Logging/status back to UI
    log_signal = pyqtSignal(str, str)

    def __init__(self, logfile=None):
        super().__init__()
        self.running = False
        self.snaptic = PySnapticSDK()
        self.logfile = logfile
        self.csv_file = None
        self.csv_writer = None

        # Rolling buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)

        # Labeling state
        self._one_shot = None
        self._last_logged_gesture = "no_gesture"

        self.gesture_once.connect(self.on_gesture_once)

    @pyqtSlot(str)
    def on_gesture_once(self, gesture):
        """Set a one-shot gesture to be used on the next CSV row only."""
        self._one_shot = gesture or "no_gesture"
        self.log_signal.emit(f"One-shot gesture armed: {self._one_shot}", "info")

    def preprocess_and_predict(self):
        """Preprocess buffer into model-ready window & predict (placeholder)."""
        if len(self.buffer) < WINDOW_SIZE:
            return None

        window = np.array(self.buffer)
        window = window - np.mean(window, axis=0, keepdims=True)

        denom = np.maximum(np.max(np.abs(window), axis=0, keepdims=True), 1e-6)
        normed = window / denom

        b, a = butter(4, 15 / (SAMPLE_RATE / 2), btype="low")
        window = filtfilt(b, a, window, axis=0)

        kernel = np.ones(5) / 5
        normed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=normed
        )

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
                "gyro_x", "gyro_y", "gyro_z", "gesture"
            ])

        try:
            start = time.time()
            while self.running and time.time() - start < 60:
                data = self.snaptic.get_imu_data()
                if data:
                    for pkt in data["packets"]:
                        acc_x = pkt["MainAccel"]["X"]
                        acc_y = pkt["MainAccel"]["Y"]
                        acc_z = pkt["MainAccel"]["Z"]
                        gyro_x = pkt["MainGyro"]["X"]
                        gyro_y = pkt["MainGyro"]["Y"]
                        gyro_z = pkt["MainGyro"]["Z"]

                        msg = (
                            f"Packet {pkt['PacketNum']} | "
                            f"Accel=({acc_x:.2f}, {acc_y:.2f}, {acc_z:.2f}) | "
                            f"Gyro=({gyro_x:.2f}, {gyro_y:.2f}, {gyro_z:.2f})"
                        )
                        self.log_signal.emit(msg, "data")

                        # Gesture logic: one-shot, then revert
                        if self._one_shot:
                            gesture_to_write = self._one_shot
                            self._one_shot = None
                            self.log_signal.emit("Gesture auto-reset to 'no_gesture'.", "info")
                        else:
                            gesture_to_write = "no_gesture"

                        if self.csv_writer:
                            self.csv_writer.writerow([
                                data["time"], pkt["PacketNum"],
                                acc_x, acc_y, acc_z,
                                gyro_x, gyro_y, gyro_z,
                                gesture_to_write
                            ])

                        # Push into buffer
                        self.buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

                        if gesture_to_write != self._last_logged_gesture:
                            self.log_signal.emit(f"CSV label now: {gesture_to_write}", "info")
                            self._last_logged_gesture = gesture_to_write

                        # Continuous prediction if buffer full
                        if len(self.buffer) == WINDOW_SIZE:
                            pred = self.preprocess_and_predict()
                            if pred:
                                self.log_signal.emit(f"Prediction: {pred}", "success")

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
        self.setGeometry(200, 200, 750, 560)

        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: goldenrod; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Top bar: gesture dropdown + one-shot button
        top_bar = QHBoxLayout()

        self.gesture_combo = QComboBox()
        self.gesture_combo.addItems(AVAILABLE_GESTURES)
        self.gesture_combo.setEditable(False)
        top_bar.addWidget(QLabel("Gesture:"))
        top_bar.addWidget(self.gesture_combo)

        self.pulse_btn = QPushButton("Do Gesture (one row)")
        self.pulse_btn.clicked.connect(self.pulse_gesture_once)
        top_bar.addWidget(self.pulse_btn)

        top_bar.addStretch(1)
        layout.addLayout(top_bar)

        # Log window
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 10))
        layout.addWidget(self.log_box)

        # Start/Stop buttons
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
        else:
            self.log("Already logging.", "info")

    def stop_logging(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested...", "info")
        else:
            self.log("Not running.", "info")

    def pulse_gesture_once(self):
        if not self.worker or not self.worker.isRunning():
            self.log("Start logging before marking gestures.", "error")
            return
        gesture = self.gesture_combo.currentText()
        self.worker.gesture_once.emit(gesture)
        self.log(f"Marking '{gesture}' for ONE row only...", "info")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapticUI()
    window.show()
    sys.exit(app.exec_())
