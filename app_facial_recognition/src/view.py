import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QInputDialog,
    QMessageBox, QApplication, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import os
import random

from recognition_facial import RecognitionFacialService


class PrincipalScreen(QWidget):
    def __init__(self, recognition_facial_service):
        super().__init__()
        self.recognition_facial_service = recognition_facial_service
        self.cap = None
        self.timer = None

        # Record State
        self.is_recording = False
        self.record_count = 0
        self.name_person_record = None
        self.dir_train_path = None
        self.detector_face = cv2.CascadeClassifier('algorithms/haarcascade_frontalface_default.xml')

        # Recognitive State
        self.is_recognitive = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Facial Recognition")
        self.setFixedSize(800, 450)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ==== Styled Buttons ====
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """

        self.button_record_face = QPushButton("📸 Record a Face")
        self.button_record_face.setStyleSheet(button_style)

        self.button_recognition = QPushButton("📁 Train Recognition")
        self.button_recognition.setStyleSheet(button_style)

        self.button_start_recognition = QPushButton("🧠 Start/Stop Recognition")
        self.button_start_recognition.setStyleSheet(button_style)

        self.button_record_face.clicked.connect(self.record_facial)
        self.button_recognition.clicked.connect(self.train_recognitive_face)
        self.button_start_recognition.clicked.connect(self.recognition)

        left_layout.addWidget(self.button_record_face)
        left_layout.addWidget(self.button_recognition)
        left_layout.addWidget(self.button_start_recognition)
        left_layout.addStretch()

        self.recognition_State = QLabel("Recognition: OFF")
        self.recognition_State.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recognition_State.setFixedHeight(30)
        self.recognition_State.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-size: 13px;
                font-weight: bold;
                padding: 5px;
                border-radius: 5px;
            }
        """)

        # ==== Webcam Preview ====
        self.webcam_preview = QLabel("Webcam Preview")
        self.webcam_preview.setBaseSize(400, 300)
        self.webcam_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_preview.setStyleSheet("""
            QLabel {
                border: 3px solid #7f8c8d;
                background-color: black;
                color: white;
                font-size: 14px;
            }
        """)

        right_layout.addWidget(self.webcam_preview)
        right_layout.addWidget(self.recognition_State)
        # ==== Set Layouts ====
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # ==== Webcam & Timer ====
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            if self.is_recording and self.record_count <= 60 or self.is_recognitive:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.detector_face.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(70, 70)
                )

                for (x, y, w, h) in detections:
                    face = self.recover_only_face(x, y, w, h, gray)
                    if self.is_recognitive:
                        self.recognition_facial_service.get_face_recognitive(face, frame, (x, y))
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        continue
                    filename = f"{self.dir_train_path}/{random.getrandbits(128)}.png"
                    cv2.imwrite(filename, face)
                    self.record_count += 1
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if not self.is_recognitive:
                    if self.record_count > 60:
                        self.is_recording = False
                        QMessageBox.information(self, "Done", "Face recording completed.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"{e}")
            # disable case Exception
            self.recognition()

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.webcam_preview.width(),
            self.webcam_preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.webcam_preview.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.timer is not None:
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def question(self, title, question):
        input_text, ok = QInputDialog.getText(self, title, question)
        if ok and input_text.strip():
            return input_text
        else:
            return None

    def record_facial(self):
        name = self.question("Name", "Write your username.")
        if name:
            try:
                dir_train = f"./faces_to_train/{name}"
                os.makedirs(dir_train, exist_ok=False)
                QMessageBox.information(self, "Recording", "Recording will start in 3 seconds. Look at the camera.")
                time.sleep(3)
                self.name_person_record = name
                self.dir_train_path = dir_train
                self.is_recording = True
                self.record_count = 0
            except OSError:
                QMessageBox.warning(self, "Warning", "This name already exists. Try another one.")
        else:
            QMessageBox.warning(self, "Warning", "You need to send a valid name.")

    def recover_only_face(self, x, y, w, h, gray_frame):
        height, width = gray_frame.shape[:2]
        x1 = max(x - 40, 0)
        y1 = max(y - 40, 0)
        x2 = min(x + w + 25, width)
        y2 = min(y + h + 25, height)
        return gray_frame[y1:y2, x1:x2]

    def train_recognitive_face(self):
        try:
            self.recognition_facial_service.train_recognitive()
            QMessageBox.information(self, "Information", "Facial Recognition trained with success.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"{e}")

    def recognition(self):
        self.is_recognitive = not self.is_recognitive
        self.button_record_face.setEnabled(not self.is_recognitive)
        status = "OFF" if not self.is_recognitive == True else "ON"
        self.recognition_State.setText(f"Recognition: {status}")


if __name__ == "__main__":
    print("Aplicação iniciando...")
    app = QApplication([])
    janela = PrincipalScreen(RecognitionFacialService())
    janela.show()
    app.exec()
