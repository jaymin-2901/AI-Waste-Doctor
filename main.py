"""
AI WASTE DOCTOR - Intelligent Waste Classification System
Main Application Launcher.
"""

import os
import sys
import traceback
from pathlib import Path

# Suppress TensorFlow C++ verbose log messages & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# TensorFlow cannot use CUDA on native Windows. Disable GPU discovery before
# TensorFlow is imported so it does not emit the native-Windows GPU warning.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

# Import application components
from config import config_manager
from ai.classifier import WasteClassifier
from camera.camera_manager import CameraManager
from ui.main_window import MainWindow
from ui.theme import ThemeManager


def global_exception_hook(exctype, value, tb):
    """Global exception handler to prevent silent crashes and alert user."""
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    print(f"[Unhandled Error]:\n{err_msg}", file=sys.stderr)
    
    # Display error message box if QApplication exists
    if QApplication.instance():
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("AI Waste Doctor Error")
        msg_box.setText("An unexpected error occurred:")
        msg_box.setInformativeText(str(value))
        msg_box.setDetailedText(err_msg)
        msg_box.exec()


def main():
    # Set sys exception hook
    sys.excepthook = global_exception_hook

    # Initialize PySide6 Application
    app = QApplication(sys.argv)
    app.setApplicationName("AI Waste Doctor")

    print("[Main] Initializing AI Waste Doctor Application...")

    # Load configuration settings
    settings = config_manager.settings
    ThemeManager.set_theme(settings.get("theme", "dark") == "dark")

    # Initialize AI Waste Classifier
    classifier = WasteClassifier(
        smoothing_frames=settings.get("smoothing_frames", 10),
        confidence_threshold=settings.get("confidence_threshold", 70.0)
    )

    if settings.get("demo_mode", False):
        classifier.demo_mode = True

    # Initialize Camera Manager
    camera_manager = CameraManager(
        camera_index=settings.get("camera_index", 0),
        mirror=settings.get("mirror_webcam", True),
        show_fps=settings.get("show_fps", True)
    )

    # Start camera capture thread on boot
    camera_manager.start_camera()

    # Create & Display Main Window
    window = MainWindow(camera_manager=camera_manager, classifier=classifier)
    window.showMaximized()

    print("[Main] Desktop application started successfully.")

    # Execute main Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
