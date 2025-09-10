import sys
import asyncio
import struct
import csv
from datetime import datetime
from bleak import BleakClient, BleakError
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QColor

DEVICE_ADDRESS = "00:80:E1:27:A2:59"
CHAR_UUID = "ddddefd0-1234-5678-9012-345678901a00"


def decode_packet(data: bytes):
    """Decode BLE payload into list of signed int16 values"""
    count = len(data) // 2
    return struct.unpack("<" + "h" * count, data)


class BleWorker(QThread):
    log_signal = pyqtSignal(str, str)  # (message, type)

    def __init__(self, address, char_uuid, logfile=None):
        super().__init__()
        self.address = address
        self.char_uuid = char_uuid
        self.client = None
        self.running = False
        self.logfile = logfile
        self.csv_writer = None
        self.csv_file = None
        self.packet_count = 0

    def handle_notify(self, sender, data: bytearray):
        """Callback when BLE notification is received"""
        values = decode_packet(data)
        self.packet_count += 1
        msg = f"[{self.packet_count:05d}] {values}"
        self.log_signal.emit(msg, "data")

        # Save to CSV
        if self.csv_writer:
            self.csv_writer.writerow(values)
            self.csv_file.flush()

    async def connect_and_listen(self):
        try:
            self.client = BleakClient(self.address)
            await self.client.connect()
            if not self.client.is_connected:
                self.log_signal.emit("failed to connect", "error")
                return

            self.log_signal.emit(f"✅ Connected to {self.address}", "success")

            # Setup CSV logging
            if self.logfile:
                self.csv_file = open(self.logfile, "w", newline="")
                self.csv_writer = csv.writer(self.csv_file)

                headers = [
                    "packet_id", "status",
                    "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "mag_x", "mag_y", "mag_z"
                ]
                # Add some flexible extras
                headers += [f"extra_{i}" for i in range(1, 21)]
                self.csv_writer.writerow(headers)

            await self.client.start_notify(self.char_uuid, self.handle_notify)
            self.log_signal.emit(f"📡 Subscribed to {self.char_uuid}", "info")

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

        # Buttons at bottom
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
        """Append colored log messages"""
        colors = {
            "success": "green",
            "error": "red",
            "info": "blue",
            "data": "black"
        }
        color = colors.get(msg_type, "black")
        self.log_box.setTextColor(QColor(color))
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

        # Status label update
        if msg_type == "success":
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif msg_type == "error":
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        elif msg_type == "info":
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: black; font-weight: bold;")
        self.status_label.setText("Status: " + msg)

    def start_ble(self):
        if self.worker is None or not self.worker.isRunning():
            logfile = f"snaptic_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.worker = BleWorker(DEVICE_ADDRESS, CHAR_UUID, logfile=logfile)
            self.worker.log_signal.connect(self.log)
            self.worker.start()
            self.log(f"Logging to {logfile}", "info")

    def stop_ble(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stopping BLE worker...", "info")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnapticUI()
    window.show()
    sys.exit(app.exec_())
