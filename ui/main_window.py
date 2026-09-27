"""
Main Application Window for AI Waste Doctor Desktop Application.
Houses header banner, navigation tabs (Upload Image, Live Camera, Experiment, Results, How It Works, Settings),
and Science Fair Display mode trigger.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QStatusBar, QMessageBox, QFrame
)

from config import config_manager
from ui.theme import ThemeManager
from ui.styles import get_style
from ui.upload_tab import UploadImageTab
from ui.live_tab import LiveClassificationTab
from ui.how_it_works_tab import HowItWorksTab
from ui.settings_tab import SettingsTab
from ui.science_fair_view import ScienceFairOverlay


class MainWindow(QMainWindow):
    """Main desktop application window."""

    def __init__(self, camera_manager, classifier, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.classifier = classifier
        self.science_fair_overlay = None

        self.setWindowTitle("AI Waste Doctor – Intelligent Waste Classification System")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(get_style())

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # Header Banner
        t = ThemeManager.get_theme()
        header_card = QFrame()
        header_card.setProperty("class", "CardFrame")
        header_card.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_card']};
                border: 1px solid {t['border']};
                border-radius: 12px;
                padding: 10px 16px;
            }}
        """)

        h_layout = QHBoxLayout(header_card)
        h_layout.setContentsMargins(8, 6, 8, 6)

        title_v = QVBoxLayout()
        title_v.setSpacing(2)

        title_lbl = QLabel("AI WASTE DOCTOR")
        title_lbl.setObjectName("AppHeaderTitle")

        subtitle_lbl = QLabel("Intelligent Waste Classification System  •  Recyclable | Dry Waste | Wet Waste")
        subtitle_lbl.setObjectName("AppHeaderSubtitle")

        title_v.addWidget(title_lbl)
        title_v.addWidget(subtitle_lbl)

        # Science Fair Mode Button
        self.btn_science_fair = QPushButton("SCIENCE FAIR MODE")
        self.btn_science_fair.setObjectName("ScienceFairBtn")
        self.btn_science_fair.setCursor(Qt.CursorShape.PointingHandCursor)

        h_layout.addLayout(title_v, 1)
        h_layout.addWidget(self.btn_science_fair)

        # Main Navigation Tabs
        self.tabs = QTabWidget()

        # 1. Upload Image Tab (primary — no lag, classify from file)
        self.tab_upload = UploadImageTab(self.classifier)
        self.tabs.addTab(self.tab_upload, "📁 UPLOAD IMAGE")

        # 2. Live Camera Tab (secondary — real-time webcam)
        self.tab_live = LiveClassificationTab(self.camera_manager, self.classifier)
        self.tabs.addTab(self.tab_live, "📹 LIVE CAMERA")

        # 3. How It Works Tab
        self.tab_how_it_works = HowItWorksTab()
        self.tabs.addTab(self.tab_how_it_works, "💡 HOW IT WORKS")

        # 4. Settings Tab
        self.tab_settings = SettingsTab(self.camera_manager, self.classifier)
        self.tabs.addTab(self.tab_settings, "⚙️ SETTINGS")

        main_layout.addWidget(header_card)
        main_layout.addWidget(self.tabs, 1)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.update_status_bar()

    def _connect_signals(self):
        self.btn_science_fair.clicked.connect(self.open_science_fair_mode)

    def update_status_bar(self):
        """Update bottom status bar indicators."""
        cam_idx = self.camera_manager.camera_index
        model_st = "DEMO MODE" if self.classifier.demo_mode else "Keras Model Loaded"
        labels_str = " | ".join(self.classifier.labels)

        msg = f"Status: {model_st}  |  Categories: {labels_str}  |  Camera: {cam_idx}"
        self.statusBar.showMessage(msg)

    def refresh_theme(self):
        """Rebuild themed widgets so inline component styles use the new palette."""
        self._build_ui()
        self._connect_signals()

    @Slot()
    def open_science_fair_mode(self):
        """Launch fullscreen Science Fair presentation mode."""
        if self.science_fair_overlay is None or not self.science_fair_overlay.isVisible():
            self.science_fair_overlay = ScienceFairOverlay(self.camera_manager, self.classifier, parent=None)
            self.science_fair_overlay.closed_signal.connect(self.on_science_fair_closed)
            self.science_fair_overlay.scan_result_captured.connect(
                self.tab_experiment.update_live_prediction
            )
            self.science_fair_overlay.showFullScreen()

    @Slot()
    def on_science_fair_closed(self):
        self.showMaximized()

    def closeEvent(self, event):
        """Clean up camera threads before closing app window."""
        self.camera_manager.stop_camera()
        if self.science_fair_overlay:
            self.science_fair_overlay.close()
        event.accept()

