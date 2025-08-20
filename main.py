import sys
import time
import socket
import threading
from collections import defaultdict, deque

import cv2 as cv
import numpy as np
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QWidget
from ultralytics import YOLO


class ViewTransformer:
    """Perspective transformer from source quadrilateral to target rectangle."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply perspective transform to (N,2) points -> (N,2)."""
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


class VideoMonitor(QWidget):
    """
    Main application window: runs YOLO tracking, visualizes results,
    estimates risk, and broadcasts status strings to TCP clients.
    """

    def __init__(self, video_source):
        super().__init__()
        self.setWindowTitle("CCTV Monitor")
        self.setGeometry(100, 100, 1800, 720)

        # YOLO model (keep weights path and settings intact)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("epoch52.pt").to(self.device)

        # TCP server settings (keep constants)
        self.server_host = "192.168.2.13"
        self.server_port = 8878
        self.clients = []
        self._start_server()

        # Areas (A, B) and drawing colors
        self.areas = {
            "B": [(1116, 52), (775, 251), (966, 337), (1244, 87)],
            "A": [(74, 85), (361, 337), (548, 251), (200, 51)],
        }
        self.RED, self.GREEN = (0, 0, 255), (0, 255, 0)
        self.T_W, self.T_H = 160, 720

        # Video capture (keep properties)
        self.cap = cv.VideoCapture(video_source)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 736)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit(1)

        # --- UI widgets ---
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)

        # Play/Pause (emergency) button
        self.is_paused = False
        self.play_pause_button = QPushButton("Emergency", self)
        self.play_pause_button.setGeometry(1300, 650, 200, 60)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # BBox show/hide toggle
        self.show_bbox = True
        self.toggle_button = QPushButton("BBox OFF", self)
        self.toggle_button.setGeometry(1550, 650, 200, 60)
        self.toggle_button.clicked.connect(self.toggle_display)

        # Titles
        self.text_table_widget = QLabel(self)
        self.text_table_widget.setGeometry(1300, 0, 300, 20)
        self.text_table_widget.setText("All Objects")

        # Table for all objects
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Confidence", "Posi(X,Y)"])
        self.table_widget.setGeometry(1300, 20, 420, 100)

        self.text_a_area_table_widget = QLabel(self)
        self.text_a_area_table_widget.setGeometry(1300, 140, 300, 20)
        self.text_a_area_table_widget.setText("Objects in Area A")

        self.a_area_table_widget = QTableWidget(self)
        self.a_area_table_widget.setColumnCount(4)
        self.a_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Posi(x,y)", "Speed(km/h)"])
        self.a_area_table_widget.setGeometry(1300, 160, 420, 100)

        self.text_b_area_table_widget = QLabel(self)
        self.text_b_area_table_widget.setGeometry(1300, 280, 300, 20)
        self.text_b_area_table_widget.setText("Objects in Area B")

        self.b_area_table_widget = QTableWidget(self)
        self.b_area_table_widget.setColumnCount(4)
        self.b_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Posi(x,y)", "Speed(km/h)"])
        self.b_area_table_widget.setGeometry(1300, 300, 420, 100)

        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 420, 500, 50)
        self.collision_label.setStyleSheet("font-size: 30px; color: red;")
        self.collision_label.setText("Risk: Safe")

        self.text_label = QLabel(self)
        self.text_label.setGeometry(1300, 500, 500, 50)
        self.text_label.setStyleSheet("font-size: 20px;")
        self.text_label.setText("Detecting objects...")

        # Timer for frame updates (keep 30 ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Caches
        self.last_frame = None
        self.last_results = None

        # Perspective transformers for A and B (keep points)
        sourceA = np.array([[74, 85], [200, 51], [361, 337], [548, 251]])
        sourceB = np.array([[1116, 52], [1244, 87], [775, 251], [966, 337]])
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)
        self.transformerB = ViewTransformer(source=sourceB, target=target)

        # Histories
        self.T_FPS = 10  # window size for speed estimation
        self.track_history = defaultdict(lambda: deque(maxlen=self.T_FPS))  # (x, y) for drawing trails
        self.area_y_hist = {"A": defaultdict(lambda: deque(maxlen=self.T_FPS)),
                            "B": defaultdict(lambda: deque(maxlen=self.T_FPS))}  # y histories per area

        # Current state for collision evaluation
        self.current_a_position = None
        self.current_b_position = None
        self.current_a_speed = None
        self.current_b_speed = None

        # Active tracks per area
        self.active_tracks = {"A": set(), "B": set()}

        # Deadlines for when objects that left the area are considered safe
        self.safe_deadlines = {"A": {}, "B": {}}

    # -------------------- UI callbacks --------------------
    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(30)
            self.play_pause_button.setText("Emergency")
        else:
            self.timer.stop()
            self.play_pause_button.setText("Cancel\nEmergency")
        self.is_paused = not self.is_paused

    def toggle_display(self):
        self.show_bbox = not self.show_bbox
        self.toggle_button.setText("BBox OFF" if self.show_bbox else "BBox ON")
        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame.copy(), self.last_results)

    # -------------------- Core loop --------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        # Run YOLO tracking (keep original settings)
        results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
        self.last_frame = frame.copy()
        self.last_results = results
        self.update_display(frame, results)

    # -------------------- Helpers --------------------
    @staticmethod
    def _calculate_speed(slope: float) -> float:
        # Keep hard-coded conversion chain
        v1 = round((slope * 0.0025 * 30), 2)
        v2 = round((v1 * 3.6), 2)
        v3 = round((v2 * 7.5), 2)
        return v3

    @staticmethod
    def _korean_to_english(status: str) -> str:
        return {"안전": "Safe", "주의": "Caution", "위험": "Danger"}.get(status, "Safe")

    def _evaluate_collision_risk(self) -> str:
        """Return risk string in Korean for downstream devices."""
        if self.current_a_position is None or self.current_b_position is None:
            return "안전"
        distance = abs(self.current_a_position - self.current_b_position)
        if distance < 120:
            return "위험"
        if distance < 350:
            return "주의"
        return "안전"

    # -------------------- Display & Logic --------------------
    def update_display(self, frame, results):
        # Reset tables
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)
        self.b_area_table_widget.setRowCount(0)

        now = time.time()

        # Draw polygons for areas
        cv.polylines(frame, [np.array(self.areas["A"], np.int32)], True, self.RED, 1)
        cv.polylines(frame, [np.array(self.areas["B"], np.int32)], True, self.RED, 1)

        # Extract YOLO outputs
        boxes_id = results and results[0].boxes.id
        if boxes_id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, box_c, tid, cls_idx, conf in zip(
                boxes_xyxy, boxes_xywh, track_ids, classes, confs
            ):
                x1, y1, x2, y2 = map(int, box)
                x, y, w, h = map(int, box_c)
                cls_name = self.model.names[int(cls_idx)]

                # Trail for visualization
                trail = self.track_history[tid]
                trail.append((x, y))
                if self.show_bbox and len(trail) > 1:
                    pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                    cv.polylines(frame, [pts], isClosed=False, color=self.GREEN, thickness=3)

                # All-objects table
                row = self.table_widget.rowCount()
                self.table_widget.insertRow(row)
                self.table_widget.setItem(row, 0, QTableWidgetItem(str(int(tid))))
                self.table_widget.setItem(row, 1, QTableWidgetItem(cls_name))
                self.table_widget.setItem(row, 2, QTableWidgetItem(f"{conf:.2f}"))
                self.table_widget.setItem(row, 3, QTableWidgetItem(f"({x}, {y})"))

                # Area membership tests
                in_A = cv.pointPolygonTest(np.array(self.areas["A"], np.int32), (x, y), False) >= 0
                in_B = cv.pointPolygonTest(np.array(self.areas["B"], np.int32), (x, y), False) >= 0

                # Transform points into rectified coordinates
                pA = self.transformerA.transform_points(np.array([[x, y]]))
                pB = self.transformerB.transform_points(np.array([[x, y]]))
                txtA = f"({int(pA[0][0])}, {int(pA[0][1])})"
                txtB = f"({int(pB[0][0])}, {int(pB[0][1])})"

                if self.show_bbox and (in_A or in_B):
                    cv.rectangle(frame, (x1, y1), (x2, y2), self.GREEN, 2)
                    cv.putText(
                        frame,
                        f"ID: {int(tid)} {cls_name}",
                        (x1, y1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.GREEN,
                        2,
                    )

                # Area A updates
                if in_A:
                    self.active_tracks["A"].add(tid)
                    yA = int(pA[0][1])
                    self.area_y_hist["A"][tid].append(yA)
                    slopeA = 0.0
                    histA = self.area_y_hist["A"][tid]
                    if len(histA) >= 2:
                        slopeA = abs(histA[-1] - histA[0]) / len(histA)
                    self.current_a_speed = self._calculate_speed(round(slopeA, 2))
                    self.current_a_position = yA

                    rowA = self.a_area_table_widget.rowCount()
                    self.a_area_table_widget.insertRow(rowA)
                    self.a_area_table_widget.setItem(rowA, 0, QTableWidgetItem(str(int(tid))))
                    self.a_area_table_widget.setItem(rowA, 1, QTableWidgetItem(cls_name))
                    self.a_area_table_widget.setItem(rowA, 2, QTableWidgetItem(txtA))
                    self.a_area_table_widget.setItem(rowA, 3, QTableWidgetItem(str(self.current_a_speed)))

                    # Clear any pending safe deadline since it's back in the area
                    self.safe_deadlines["A"].pop(tid, None)

                elif tid in self.active_tracks["A"] and tid not in self.safe_deadlines["A"]:
                    # Mark when this object will be considered safe after leaving A
                    time_to_safe = self._time_to_safe(self.current_a_speed)
                    self.safe_deadlines["A"][tid] = now + time_to_safe

                # Area B updates
                if in_B:
                    self.active_tracks["B"].add(tid)
                    yB = int(pB[0][1])
                    self.area_y_hist["B"][tid].append(yB)
                    slopeB = 0.0
                    histB = self.area_y_hist["B"][tid]
                    if len(histB) >= 2:
                        slopeB = abs(histB[-1] - histB[0]) / len(histB)
                    self.current_b_speed = self._calculate_speed(round(slopeB, 2))
                    self.current_b_position = yB

                    rowB = self.b_area_table_widget.rowCount()
                    self.b_area_table_widget.insertRow(rowB)
                    self.b_area_table_widget.setItem(rowB, 0, QTableWidgetItem(str(int(tid))))
                    self.b_area_table_widget.setItem(rowB, 1, QTableWidgetItem(cls_name))
                    self.b_area_table_widget.setItem(rowB, 2, QTableWidgetItem(txtB))
                    self.b_area_table_widget.setItem(rowB, 3, QTableWidgetItem(str(self.current_b_speed)))

                    self.safe_deadlines["B"].pop(tid, None)

                elif tid in self.active_tracks["B"] and tid not in self.safe_deadlines["B"]:
                    time_to_safe = self._time_to_safe(self.current_b_speed)
                    self.safe_deadlines["B"][tid] = now + time_to_safe

        # Clear tracks that have been safe long enough
        self._cleanup_safe(now)

        # Risk estimation and broadcast (keep thresholds/logic)
        status_ko = self._evaluate_collision_risk()
        # Only consider collisions once positions are sufficiently advanced in the frame
        if (self.current_a_position is not None and self.current_a_position >= 450) or (
            self.current_b_position is not None and self.current_b_position >= 450
        ):
            status_ko = self._evaluate_collision_risk()

        # Update UI label in English; broadcast Korean for devices
        self.collision_label.setText(f"Risk: {self._korean_to_english(status_ko)}")
        self._broadcast_message(status_ko)

        # Render
        frame = cv.resize(frame, (1280, 720))
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    # -------------------- Area exit helpers --------------------
    @staticmethod
    def _time_to_safe(speed: float) -> float:
        """
        Map speed (km/h) to seconds to wait before considering 'safe' after exit.
        Keep original piecewise-linear mapping.
        """
        max_speed = 40  # fastest
        min_speed = 10  # slowest
        min_time = 1    # 1s at max speed
        max_time = 5    # 5s at min speed
        if speed is None:
            return max_time
        if speed >= max_speed:
            return min_time
        if speed <= min_speed:
            return max_time
        ratio = (max_speed - speed) / (max_speed - min_speed)
        return round(min_time + (max_time - min_time) * ratio, 2)

    def _cleanup_safe(self, now: float) -> None:
        """Remove tracks whose safe deadlines have passed and reset positions/speeds when empty."""
        for area in ("A", "B"):
            expired = [tid for tid, t in self.safe_deadlines[area].items() if now - t > 0]
            for tid in expired:
                self.active_tracks[area].discard(tid)
                self.safe_deadlines[area].pop(tid, None)
                # Also clear history for that id
                self.area_y_hist[area].pop(tid, None)

        # Reset current positions if no active tracks remain in area
        if not self.active_tracks["A"]:
            self.current_a_position = None
            self.current_a_speed = None
        if not self.active_tracks["B"]:
            self.current_b_position = None
            self.current_b_speed = None

    # -------------------- Networking --------------------
    def _start_server(self):
        t = threading.Thread(target=self._server_thread, daemon=True)
        t.start()

    def _server_thread(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.server_host, self.server_port))
        server.listen(5)
        print(f"[LISTENING] Server is listening on {self.server_host}:{self.server_port}")
        while True:
            client_socket, addr = server.accept()
            threading.Thread(target=self._handle_client, args=(client_socket, addr), daemon=True).start()

    def _broadcast_message(self, message: str):
        """Broadcast message to all clients (append newline for line-based readers)."""
        msg = (message + "\n").encode("utf-8")
        for client in list(self.clients):
            try:
                client.sendall(msg)
            except Exception:
                try:
                    client.close()
                finally:
                    if client in self.clients:
                        self.clients.remove(client)

    def _handle_client(self, client_socket, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        self.clients.append(client_socket)
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
        except Exception:
            pass
        finally:
            print(f"[DISCONNECTED] {addr} disconnected.")
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            try:
                client_socket.close()
            except Exception:
                pass

    # -------------------- Lifecycle --------------------
    def closeEvent(self, event):
        try:
            self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_source = 0
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())
