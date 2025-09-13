import sys
import asyncio
import struct
import csv
import time
from datetime import datetime
import numpy as np
from bleak import BleakClient, BleakError
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from collections import deque
from scipy.signal import butter, filtfilt


# Config values
WINDOW_SECONDS = 2.56          # window length
SAMPLE_RATE = 48               # Hz
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLE_RATE)
DEVICE_ADDRESS = "00:80:E1:27:A2:59"
CHAR_UUID = "ddddefd0-1234-5678-9012-345678901a00"

def decode_packet(data: bytes):
    """Decode BLE payload into list of signed int16 values."""
    count = len(data) // 2
    return struct.unpack("<" + "h" * count, data)

class BleWorker(QThread):
    log_signal = pyqtSignal(str, str)

    def __init__(self, address, char_uuid, logfile=None):
        super().__init__()
        self.address = address
        self.char_uuid = char_uuid
        self.client = None
        self.running = False
        self.logfile = logfile
        self.csv_writer = None
        self.csv_file = None

        self.packet_count = 0              # 1,2,3
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.last_time = None              # wall-clock for Δt debug
        self.start_time_wall = None        # wall-clock t0
        self.label = "unlabeled"

    def set_label(self, label: str):
        self.label = label

    def handle_notify(self, sender, data: bytearray):
        values = decode_packet(data)
        self.packet_count += 1
        now = time.time()

        # Initialize t0 on first packet
        if self.start_time_wall is None:
            self.start_time_wall = now

        # Δt debug 
        if self.last_time is not None:
            dt = now - self.last_time
            self.log_signal.emit(f"Δt: {dt:.4f} sec", "info")
        self.last_time = now

        #  sample index for elapsed time 
        sample_index = self.packet_count - 1           # 0-based
        elapsed_nominal = sample_index / SAMPLE_RATE   # seconds

        # Wall-clock elapsed
        elapsed_wall = now - self.start_time_wall

        # Show packet in UI
        self.log_signal.emit(f"[{self.packet_count:05d}] {values}", "data")

        # Save to CSV 
        if self.csv_writer:
            clock_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            row = [
                f"{elapsed_nominal:.4f}",    # ideal timeline
                f"{elapsed_wall:.4f}",       # wall-clock timeline
                clock_ts,                    # readable clock in time seconds
                self.packet_count
            ] + list(values) + [self.label]
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # Buffer accel only 
        try:
            acc_x, acc_y, acc_z = values[2], values[3], values[4]
            self.buffer.append([acc_x, acc_y, acc_z])
        except Exception as e:
            self.log_signal.emit(f"Buffering error: {e}", "error")

        # pass to the data for ml prediction when the buffer is full
        # handles the data the same way in the prediction file 
        if len(self.buffer) == WINDOW_SIZE:
            window = np.array(self.buffer)                       # (N,3)
            # mean removal then per-channel normalization to [-1,1]
            window = window - np.mean(window, axis=0, keepdims=True)
            denom = np.maximum(np.max(np.abs(window), axis=0, keepdims=True), 1e-6)
            normed = window / denom
            
            b, a = butter(4, 15/(SAMPLE_RATE/2), btype='low')   # 15 Hz cutoff
            window = filtfilt(b, a, window, axis=0)

            kernel = np.ones(5)/5
            normed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=normed)
            
            
            
            # TODO: model will be built and placed here for predictin and ui improvement



            fake_pred = np.random.choice(["walking", "running", "idle"])
            self.log_signal.emit(f"Prediction: {fake_pred}", "success")

            # 50% overlap
            for _ in range(WINDOW_SIZE // 2):
                if self.buffer:
                    self.buffer.popleft()

    async def connect_and_listen(self):
        try:
            self.client = BleakClient(self.address)
            await self.client.connect()
            if not self.client.is_connected:
                self.log_signal.emit("failed to connect", "error")
                return

            self.log_signal.emit(f"Connected to {self.address}", "success")

            # CSV setup
            if self.logfile:
                self.csv_file = open(self.logfile, "w", newline="")
                self.csv_writer = csv.writer(self.csv_file)
                headers = [
                    "elapsed_nominal_s",      # from sample index
                    "elapsed_wall_s",         # from wall-clock
                    "timestamp",              # HH:MM:SS.mmm
                    "packet_id",
                    "status",
                    "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "mag_x", "mag_y", "mag_z",
                ]
                headers += [f"extra_{i}" for i in range(1, 21)]
                headers.append("label")
                self.csv_writer.writerow(headers)

            await self.client.start_notify(self.char_uuid, self.handle_notify)
            self.log_signal.emit(f"Subscribed to {self.char_uuid}", "info")

            while self.running:
                await asyncio.sleep(0.5)

            await self.client.stop_notify(self.char_uuid)
            await self.client.disconnect()
            self.log_signal.emit("Disconnected", "error")

        except BleakError as e:
            self.log_signal.emit(f"BLE Error: {e}", "error")
        finally:
            if self.csv_file:
                self.csv_file.close()

    def run(self):
        self.running = True
        asyncio.run(self.connect_and_listen())

    def stop(self):
        self.running = False

class SnapticUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snaptic IMU BLE Logger")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # Status bar
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: goldenrod; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 10))
        layout.addWidget(self.log_box)

        # Activity label selector
        self.label_box = QComboBox()
        self.label_box.addItems(["unlabeled", "walking", "running", "idle"])
        layout.addWidget(QLabel("Select Activity Label:"))
        layout.addWidget(self.label_box)
        self.label_box.currentTextChanged.connect(self.update_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect & Start Logging")
        self.connect_btn.clicked.connect(self.start_ble)
        btn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Stop & Disconnect")
        self.disconnect_btn.clicked.connect(self.stop_ble)
        btn_layout.addWidget(self.disconnect_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.worker = None

    def log(self, msg: str, msg_type: str = "info"):
        colors = {"success": "green", "error": "red", "info": "blue", "data": "black"}
        self.log_box.setTextColor(QColor(colors.get(msg_type, "black")))
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

        css = {"success": "green", "error": "red", "info": "blue"}.get(msg_type, "black")
        self.status_label.setStyleSheet(f"color: {css}; font-weight: bold;")
        self.status_label.setText("Status: " + msg)
    # start the bloutouth 
    def start_ble(self):
        if self.worker is None or not self.worker.isRunning():
            logfile = f"snaptic_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.worker = BleWorker(DEVICE_ADDRESS, CHAR_UUID, logfile=logfile)
            self.worker.log_signal.connect(self.log)
            self.worker.set_label(self.label_box.currentText())
            self.worker.start()
            self.log(f"Logging to {logfile}", "info")
    # updating the label which is last row
    def update_label(self, label):
        if self.worker:
            self.worker.set_label(label)
        self.log(f"Label set to: {label}", "info")
    # stop connection
    def stop_ble(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stopping BLE worker...", "info")

class BLEShim:
    def __init__(self, sample_rate=48):
        self.sample_rate = sample_rate
        self.buffer = []

    def push_packet(self, values):
        self.buffer.append(values)

    def get_current_board_data(self, num_points):
        # Return last num_points as numpy array
        import numpy as np
        data = np.array(self.buffer[-num_points:]).T  # shape [channels, samples]
        return data

if __name__ == "__main__":
    # Test BLEShim before starting Qt
    ble_shim = BLEShim(sample_rate=48)
    for i in range(200):
        ble_shim.push_packet([i, i*2, i*3])  # fake acc_x, acc_y, acc_z
    data = ble_shim.get_current_board_data(128)
    print("Test data shape:", data.shape)
    print(data[:, :5])  # first 5 samples

    # Start Qt app
    app = QApplication(sys.argv)
    window = SnapticUI()
    window.show()
    sys.exit(app.exec_())