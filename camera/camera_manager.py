"""
Camera Manager module for AI Waste Doctor.
Runs video capture in a dedicated PySide6 QThread, draws scan zone overlay,
calculates FPS, handles camera mirroring, and provides non-blocking frame signals.
"""

import time
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from PySide6.QtCore import QThread, Signal, Slot, QObject
from PySide6.QtGui import QImage


def cv_frame_to_qimage(frame):
    """Convert an OpenCV BGR frame to PySide6 QImage."""
    if frame is None:
        return QImage()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    # Convert BGR to RGB for PySide6
    if CV2_AVAILABLE:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame[..., ::-1]
    return QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)


class CameraWorker(QObject):
    """Worker object that runs inside a QThread to capture webcam frames continuously."""

    frame_ready = Signal(object, object, tuple, float, bool, str)
    # Signal signature: (qimage, raw_bgr_frame, scan_zone_box, fps, connected, status_text)

    def __init__(self, camera_index=0, mirror=True, show_fps=True, scan_zone_ratio=0.6):
        super().__init__()
        self.camera_index = camera_index
        self.mirror = mirror
        self.show_fps = show_fps
        self.scan_zone_ratio = max(0.2, min(float(scan_zone_ratio), 0.95))
        
        self.running = False
        self.cap = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.connected = False
        self.status_text = "Disconnected"

    @Slot()
    def start_capture(self):
        """Open camera and enter frame loop."""
        self.running = True
        self._init_camera()

        while self.running:
            if self.connected and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Apply horizontal mirror if enabled
                    if self.mirror and CV2_AVAILABLE:
                        frame = cv2.flip(frame, 1)

                    h, w = frame.shape[:2]

                    scan_zone_box = self.get_scan_zone(w, h)
                    box_x, box_y, box_w, box_h = scan_zone_box

                    # Calculate real-time FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.start_time = time.time()

                    # Copy frame for drawing overlays
                    annotated_frame = frame.copy()
                    if CV2_AVAILABLE:
                        # Draw Rectangular Scan Zone (Emerald Green #10b981 / BGR: 129, 185, 16)
                        color = (129, 185, 16)
                        cv2.rectangle(annotated_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 3)

                        # Draw subtle corner brackets for sci-fi look
                        bracket_len = 25
                        # Top-Left corner
                        cv2.line(annotated_frame, (box_x, box_y), (box_x + bracket_len, box_y), (255, 255, 255), 4)
                        cv2.line(annotated_frame, (box_x, box_y), (box_x, box_y + bracket_len), (255, 255, 255), 4)
                        # Top-Right corner
                        cv2.line(annotated_frame, (box_x + box_w, box_y), (box_x + box_w - bracket_len, box_y), (255, 255, 255), 4)
                        cv2.line(annotated_frame, (box_x + box_w, box_y), (box_x + box_w, box_y + bracket_len), (255, 255, 255), 4)
                        # Bottom-Left corner
                        cv2.line(annotated_frame, (box_x, box_y + box_h), (box_x + bracket_len, box_y + box_h), (255, 255, 255), 4)
                        cv2.line(annotated_frame, (box_x, box_y + box_h), (box_x, box_y + box_h - bracket_len), (255, 255, 255), 4)
                        # Bottom-Right corner
                        cv2.line(annotated_frame, (box_x + box_w, box_y + box_h), (box_x + box_w - bracket_len, box_y + box_h), (255, 255, 255), 4)
                        cv2.line(annotated_frame, (box_x + box_w, box_y + box_h), (box_x + box_w, box_y + box_h - bracket_len), (255, 255, 255), 4)

                        # Text: PLACE OBJECT INSIDE SCAN ZONE
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = "PLACE OBJECT INSIDE SCAN ZONE"
                        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
                        text_x = box_x + (box_w - text_size[0]) // 2
                        text_y = box_y - 12 if box_y - 12 > 20 else box_y + 30
                        
                        # Black text shadow for readability
                        cv2.putText(annotated_frame, text, (text_x + 1, text_y + 1), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(annotated_frame, text, (text_x, text_y), font, 0.6, (50, 230, 150), 2, cv2.LINE_AA)

                        # FPS Overlay in top-right corner if enabled
                        if self.show_fps:
                            fps_text = f"FPS: {self.fps:.1f}"
                            cv2.putText(annotated_frame, fps_text, (w - 110, 30), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(annotated_frame, fps_text, (w - 111, 29), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                    qimg = cv_frame_to_qimage(annotated_frame)
                    self.frame_ready.emit(qimg, frame, scan_zone_box, self.fps, True, "Camera Connected")
                    time.sleep(0.03)  # Target ~30 FPS frame yield
                else:
                    self.connected = False
                    self.status_text = "Camera Read Error"
                    self._emit_placeholder_frame("CAMERA DISCONNECTED")
                    time.sleep(0.5)
            else:
                self._emit_placeholder_frame("NO CAMERA DETECTED")
                time.sleep(0.5)

        self._close_camera()

    def get_scan_zone(self, width, height):
        """Return a centered square scan zone for the camera frame."""
        box_size = int(min(width, height) * self.scan_zone_ratio)
        box_x = (width - box_size) // 2
        box_y = (height - box_size) // 2
        return box_x, box_y, box_size, box_size

    def _init_camera(self):
        """Attempt opening camera device."""
        if not CV2_AVAILABLE:
            self.connected = False
            self.status_text = "OpenCV (cv2) Not Installed"
            return

        try:
            if self.cap is not None:
                self.cap.release()

            # Try default backend first (most compatible on Windows)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

            if not self.cap.isOpened():
                # Fallback: try DirectShow backend
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            if self.cap.isOpened():
                # Set explicit resolution for stable capture
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                # Warm up: discard first few frames (often garbled on Windows)
                for _ in range(5):
                    self.cap.read()
                self.connected = True
                self.status_text = "Camera Connected"
            else:
                self.connected = False
                self.status_text = "NO CAMERA DETECTED"
        except Exception as e:
            self.connected = False
            self.status_text = f"Camera Error: {e}"

    def _close_camera(self):
        """Release camera resource."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.connected = False

    def _emit_placeholder_frame(self, message="NO CAMERA DETECTED"):
        """Emit a dark styled placeholder image when camera is unavailable."""
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Fill dark navy gradient-like background
        img[:, :] = (30, 20, 15)  # Dark BGR

        if CV2_AVAILABLE:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Camera icon/symbol text
            cv2.putText(img, "📷 " + message, (130, 220), font, 0.8, (100, 100, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Please connect a webcam or check Camera Index in Settings", (70, 270), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        qimg = cv_frame_to_qimage(img)
        dummy_box = (100, 80, 440, 320)
        self.frame_ready.emit(qimg, img, dummy_box, 0.0, False, message)

    @Slot()
    def stop_capture(self):
        """Signal frame loop to stop."""
        self.running = False


class CameraManager(QObject):
    """Wrapper class managing the CameraWorker thread lifecycle for PySide6 GUI."""

    frame_signal = Signal(object, object, tuple, float, bool, str)

    def __init__(self, camera_index=0, mirror=True, show_fps=True, scan_zone_ratio=0.6):
        super().__init__()
        self.camera_index = camera_index
        self.mirror = mirror
        self.show_fps = show_fps
        self.scan_zone_ratio = scan_zone_ratio
        
        self.thread = None
        self.worker = None

    def start_camera(self):
        """Initialize worker thread and start camera capture."""
        if self.thread is not None and self.thread.isRunning():
            self.stop_camera()

        self.thread = QThread()
        self.worker = CameraWorker(
            camera_index=self.camera_index,
            mirror=self.mirror,
            show_fps=self.show_fps,
            scan_zone_ratio=self.scan_zone_ratio
        )
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.start_capture)
        self.worker.frame_ready.connect(self.frame_signal.emit)

        self.thread.start()

    def stop_camera(self):
        """Gracefully stop camera worker and terminate thread."""
        if self.worker is not None:
            self.worker.stop_capture()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)
            self.thread = None
            self.worker = None

    def set_camera_index(self, index: int):
        """Update active camera index and restart stream."""
        self.camera_index = index
        self.start_camera()

    def set_mirror(self, mirror: bool):
        """Toggle frame horizontal flip."""
        self.mirror = mirror
        if self.worker:
            self.worker.mirror = mirror

    def set_show_fps(self, show_fps: bool):
        """Toggle FPS display overlay."""
        self.show_fps = show_fps
        if self.worker:
            self.worker.show_fps = show_fps

    def set_scan_zone_ratio(self, ratio: float):
        """Update the centered square scan-zone size."""
        self.scan_zone_ratio = max(0.2, min(float(ratio), 0.95))
        if self.worker:
            self.worker.scan_zone_ratio = self.scan_zone_ratio
