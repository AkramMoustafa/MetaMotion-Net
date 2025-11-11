import sys
import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg

from snaptic_sdk import PySnapticSDK

SAMPLE_RATE = 48
WINDOW_SECONDS = 2.65
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)

class StreamWorker(QThread):
    data_signal = pyqtSignal(np.ndarray)
    pred_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.snaptic = PySnapticSDK()
        self.buffer = deque(maxlen=WINDOW_SIZE)

    def preprocess_and_predict(self, window):
        # Same preprocessing as logger
        window = window - np.mean(window, axis=0, keepdims=True)
        denom = np.maximum(np.max(np.abs(window), axis=0, keepdims=True), 1e-6)
        normed = window / denom

        b, a = butter(4, 15 / (SAMPLE_RATE / 2), btype="low")
        window = filtfilt(b, a, normed, axis=0)

        # FAKE PREDICTION (replace with trained model later)
        return np.random.choice(["no_gesture", "swipe_up", "swipe_left", "swipe_right"])

    def run(self):
        devices = self.snaptic.search_devices()
        if not devices:
            self.pred_signal.emit("No device found.")
            return
        device = devices[0]
        if not self.snaptic.connect_device(device):
            self.pred_signal.emit("Connection failed.")
            return

        self.running = True
        while self.running:
            data = self.snaptic.get_imu_data()
            if data:
                for pkt in data["packets"]:
                    acc_x = pkt["MainAccel"]["X"]
                    acc_y = pkt["MainAccel"]["Y"]
                    acc_z = pkt["MainAccel"]["Z"]

                    gyro_x = pkt["MainGyro"]["X"]
                    gyro_y = pkt["MainGyro"]["Y"]
                    gyro_z = pkt["MainGyro"]["Z"]

                    sample = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                    self.buffer.append(sample)

                    if len(self.buffer) == WINDOW_SIZE:
                        arr = np.array(self.buffer)
                        pred = self.preprocess_and_predict(arr)
                        self.pred_signal.emit(pred)
                        self.data_signal.emit(arr[-200:])  # last ~4 seconds
            time.sleep(0.02)

    def stop(self):
        self.running = False
        self.snaptic.disconnect_device()


class StreamUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Gesture Stream")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-2000, 2000)
        layout.addWidget(self.plot_widget)

        self.curves = [self.plot_widget.plot(pen=pg.mkPen(c)) for c in ['r','g','b','y','m','c']]
        
        self.pred_label = QLabel("Prediction: None")
        self.pred_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.pred_label)

        self.setLayout(layout)

        self.worker = StreamWorker()
        self.worker.data_signal.connect(self.update_plot)
        self.worker.pred_signal.connect(self.update_prediction)
        self.worker.start()

    def update_plot(self, arr):
        for i, curve in enumerate(self.curves):
            curve.setData(arr[:, i])

    def update_prediction(self, pred):
        self.pred_label.setText(f"Prediction: {pred}")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StreamUI()
    window.show()
    sys.exit(app.exec_())
