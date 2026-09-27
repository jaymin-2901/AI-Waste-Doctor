"""
Settings Tab for AI Waste Doctor.
Allows students and developers to adjust confidence thresholds, smoothing frames, camera indices,
normalization modes, webcam mirroring, FPS overlays, and DEMO MODE settings.
"""

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton, QFrame,
    QMessageBox, QScrollArea
)

from config import config_manager
from ui.theme import ThemeManager
from ui.styles import get_style


class SettingsTab(QWidget):
    """Configuration & Settings UI panel."""

    def __init__(self, camera_manager, classifier, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.classifier = classifier

        self._build_ui()
        self.load_current_settings()
        self._connect_signals()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)

        header_lbl = QLabel("SYSTEM CONFIGURATION & SETTINGS")
        header_lbl.setObjectName("H2")

        card = QFrame()
        card.setProperty("class", "CardFrame")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(16)
        
        t = ThemeManager.get_theme()

        # Theme Selection
        layout_theme = QHBoxLayout()
        lbl_theme = QLabel("Application Theme:")
        lbl_theme.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("Dark Mode (Recommended)", "dark")
        self.combo_theme.addItem("Light Mode", "light")
        layout_theme.addWidget(lbl_theme)
        layout_theme.addStretch()
        layout_theme.addWidget(self.combo_theme)

        # 1. Confidence Threshold
        layout_thresh = QHBoxLayout()
        lbl_thresh = QLabel("Confidence Threshold (%):")
        lbl_thresh.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(50.0, 99.0)
        self.spin_threshold.setSingleStep(1.0)
        self.spin_threshold.setValue(70.0)
        layout_thresh.addWidget(lbl_thresh)
        layout_thresh.addStretch()
        layout_thresh.addWidget(self.spin_threshold)

        # 2. Prediction Smoothing Frames
        layout_smooth = QHBoxLayout()
        lbl_smooth = QLabel("Prediction Smoothing Frames (Filter Window):")
        lbl_smooth.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 30)
        self.spin_smooth.setValue(10)
        layout_smooth.addWidget(lbl_smooth)
        layout_smooth.addStretch()
        layout_smooth.addWidget(self.spin_smooth)

        # 3. Camera Index
        layout_cam = QHBoxLayout()
        lbl_cam = QLabel("Webcam Device Index:")
        lbl_cam.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.spin_cam_idx = QSpinBox()
        self.spin_cam_idx.setRange(0, 5)
        self.spin_cam_idx.setValue(0)
        layout_cam.addWidget(lbl_cam)
        layout_cam.addStretch()
        layout_cam.addWidget(self.spin_cam_idx)

        # 4. Normalization Mode
        layout_norm = QHBoxLayout()
        lbl_norm = QLabel("Image Normalization Preprocessing Mode:")
        lbl_norm.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.combo_norm = QComboBox()
        self.combo_norm.addItem("0 to 1 (pixel / 255.0)", "0-1")
        self.combo_norm.addItem("-1 to 1 (Teachable Machine: (pixel / 127.5) - 1.0)", "-1_to_1")
        layout_norm.addWidget(lbl_norm)
        layout_norm.addStretch()
        layout_norm.addWidget(self.combo_norm)

        # 5. Default Startup Mode
        layout_def_mode = QHBoxLayout()
        lbl_def_mode = QLabel("Default Scan Mode on Startup:")
        lbl_def_mode.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.combo_def_mode = QComboBox()
        self.combo_def_mode.addItem("Manual Button Scan (Only scans when button pressed - No Lag)", "manual")
        self.combo_def_mode.addItem("Live Auto-Detect (Continuous background scanning)", "live")
        layout_def_mode.addWidget(lbl_def_mode)
        layout_def_mode.addStretch()
        layout_def_mode.addWidget(self.combo_def_mode)

        # 6. AI Inference Throttle Interval
        layout_interval = QHBoxLayout()
        lbl_interval = QLabel("Live AI Inference Frequency (Lag Prevention):")
        lbl_interval.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 20)
        self.spin_interval.setValue(5)
        self.spin_interval.setSuffix(" frames")
        layout_interval.addWidget(lbl_interval)
        layout_interval.addStretch()
        layout_interval.addWidget(self.spin_interval)

        # 7. Checkboxes
        self.chk_mirror = QCheckBox("Mirror Webcam Feed Horizontally")
        self.chk_mirror.setChecked(True)

        self.chk_fps = QCheckBox("Show Real-Time FPS Overlay")
        self.chk_fps.setChecked(True)

        self.chk_sound = QCheckBox("Enable Audio Sound Bleeps (Optional)")
        self.chk_sound.setChecked(False)

        self.chk_demo = QCheckBox("Force DEMO MODE (Simulated Predictions)")
        self.chk_demo.setChecked(False)

        # Save Settings Button
        self.btn_save = QPushButton("💾 SAVE CONFIGURATION & RESTART")
        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['accent_primary']};
                font-size: 14px;
                font-weight: 800;
                padding: 12px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {t['accent_secondary']};
            }}
        """)

        card_layout.addLayout(layout_theme)
        card_layout.addLayout(layout_thresh)
        card_layout.addLayout(layout_smooth)
        card_layout.addLayout(layout_cam)
        card_layout.addLayout(layout_norm)
        card_layout.addLayout(layout_def_mode)
        card_layout.addLayout(layout_interval)
        card_layout.addWidget(self.chk_mirror)
        card_layout.addWidget(self.chk_fps)
        card_layout.addWidget(self.chk_sound)
        card_layout.addWidget(self.chk_demo)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.btn_save)

        scroll_layout.addWidget(header_lbl)
        scroll_layout.addWidget(card)
        scroll.setWidget(scroll_widget)

        main_layout.addWidget(scroll)

    def load_current_settings(self):
        """Populate settings controls from ConfigManager."""
        conf = config_manager.settings

        theme = conf.get("theme", "dark")
        idx_t = self.combo_theme.findData(theme)
        if idx_t >= 0:
            self.combo_theme.setCurrentIndex(idx_t)

        self.spin_threshold.setValue(conf.get("confidence_threshold", 70.0))
        self.spin_smooth.setValue(conf.get("smoothing_frames", 10))
        self.spin_cam_idx.setValue(conf.get("camera_index", 0))

        norm_mode = conf.get("normalization_mode", "0-1")
        idx = self.combo_norm.findData(norm_mode)
        if idx >= 0:
            self.combo_norm.setCurrentIndex(idx)

        def_mode = conf.get("default_scan_mode", "manual")
        idx_m = self.combo_def_mode.findData(def_mode)
        if idx_m >= 0:
            self.combo_def_mode.setCurrentIndex(idx_m)

        self.spin_interval.setValue(conf.get("inference_interval_frames", 5))

        self.chk_mirror.setChecked(conf.get("mirror_webcam", True))
        self.chk_fps.setChecked(conf.get("show_fps", True))
        self.chk_sound.setChecked(conf.get("sound_enabled", False))
        self.chk_demo.setChecked(conf.get("demo_mode", False))

    def _connect_signals(self):
        self.btn_save.clicked.connect(self.on_save_clicked)

    @Slot()
    def on_save_clicked(self):
        """Save settings to config.json and apply to running components."""
        new_theme = self.combo_theme.currentData()
        new_thresh = self.spin_threshold.value()
        new_smooth = self.spin_smooth.value()
        new_cam_idx = self.spin_cam_idx.value()
        new_norm = self.combo_norm.currentData()
        new_def_mode = self.combo_def_mode.currentData()
        new_interval = self.spin_interval.value()
        new_mirror = self.chk_mirror.isChecked()
        new_fps = self.chk_fps.isChecked()
        new_sound = self.chk_sound.isChecked()
        new_demo = self.chk_demo.isChecked()

        config_manager.set("theme", new_theme)
        config_manager.set("confidence_threshold", new_thresh)
        config_manager.set("smoothing_frames", new_smooth)
        config_manager.set("camera_index", new_cam_idx)
        config_manager.set("normalization_mode", new_norm)
        config_manager.set("default_scan_mode", new_def_mode)
        config_manager.set("inference_interval_frames", new_interval)
        config_manager.set("mirror_webcam", new_mirror)
        config_manager.set("show_fps", new_fps)
        config_manager.set("sound_enabled", new_sound)
        config_manager.set("demo_mode", new_demo)

        # Apply the selected theme to the running application immediately.
        ThemeManager.set_theme(new_theme == "dark")
        main_window = self.window()
        main_window.setStyleSheet(get_style(new_theme == "dark"))
        QTimer.singleShot(0, main_window.refresh_theme)

        # Apply immediately to live classifier & camera manager
        self.classifier.set_confidence_threshold(new_thresh)
        self.classifier.set_smoothing_frames(new_smooth)
        self.classifier.demo_mode = new_demo

        if self.camera_manager.camera_index != new_cam_idx:
            self.camera_manager.set_camera_index(new_cam_idx)
        self.camera_manager.set_mirror(new_mirror)
        self.camera_manager.set_show_fps(new_fps)
        
        msg = "Application settings saved and applied successfully."

        QMessageBox.information(self, "Settings Saved", msg)
