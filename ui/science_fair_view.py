"""
Science Fair Display Mode View for AI Waste Doctor.
Full-screen presentation view optimized for exhibition booths, judges, and visitors standing feet away.
Features giant typography, high-contrast video, prominent prediction displays, and Escape key exit handler.
"""

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QFont, QKeyEvent, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)

from config import CATEGORY_GUIDANCE, GENERIC_GUIDANCE, config_manager
from ui.theme import ThemeManager
from ui.widgets import ScanZoneVideoWidget, ConfidenceProgressBar, CategoryGuidanceCard
from ui.feedback import play_decision_buzzer


class ScienceFairOverlay(QWidget):
    """Full-screen display window for Science Fair Booth Presentation."""

    closed_signal = Signal()
    scan_result_captured = Signal(str, float)

    def __init__(self, camera_manager, classifier, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.classifier = classifier
        self.confidence_bars = {}
        self.frame_counter = 0
        self.candidate_class = None
        self.candidate_count = 0
        self.final_decision_locked = False

        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {t['bg_main']};
                color: {t['text_primary']};
            }}
        """)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Top Exhibition Header
        header_card = QFrame()
        t = ThemeManager.get_theme()
        header_card.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_card']};
                border: 2px solid {t['accent_primary']};
                border-radius: 16px;
                padding: 12px;
            }}
        """)
        h_layout = QHBoxLayout(header_card)

        titles_v = QVBoxLayout()
        titles_v.setSpacing(2)

        main_title = QLabel("AI WASTE DOCTOR")
        main_title.setStyleSheet(f"font-size: 32px; font-weight: 900; color: {t['accent_primary']}; letter-spacing: 2px;")

        subtitle = QLabel("SCAN  ➔  CLASSIFY  ➔  SORT  ➔  CLEANER PLANET")
        subtitle.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {t['accent_secondary']}; letter-spacing: 1px;")

        titles_v.addWidget(main_title)
        titles_v.addWidget(subtitle)

        # Science Tag & Exit Button
        tag_lbl = QLabel("🌱 AI for Better Life")
        tag_lbl.setStyleSheet(f"""
            background-color: {t['bg_secondary']};
            color: {t['accent_primary']};
            font-weight: 800;
            font-size: 14px;
            padding: 8px 16px;
            border-radius: 16px;
        """)

        self.btn_exit = QPushButton("❌ EXIT (ESC)")
        self.btn_exit.setObjectName("SecondaryButton")
        self.btn_exit.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['bg_secondary']};
                color: {t['text_primary']};
                font-size: 13px;
                font-weight: 800;
                padding: 10px 18px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {t['error']};
            }}
        """)

        self.btn_reset = QPushButton("↻ SCAN NEXT OBJECT")
        self.btn_reset.setObjectName("SecondaryButton")
        self.btn_reset.clicked.connect(self.reset_decision)

        h_layout.addLayout(titles_v, 1)
        h_layout.addWidget(tag_lbl)
        h_layout.addSpacing(16)
        h_layout.addWidget(self.btn_reset)
        h_layout.addWidget(self.btn_exit)

        # Center Grid: Left Massive Camera + Right Huge Prediction Box
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(20)

        # Left Video
        self.video_widget = ScanZoneVideoWidget()
        self.video_widget.setMinimumSize(640, 480)

        # Right AI Prediction Panel
        right_card = QFrame()
        right_card.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_card']};
                border: 2px solid {t['border']};
                border-radius: 16px;
                padding: 20px;
            }}
        """)
        r_layout = QVBoxLayout(right_card)
        r_layout.setSpacing(16)

        p_title = QLabel("REAL-TIME AI CLASSIFICATION")
        p_title.setStyleSheet(f"font-size: 18px; font-weight: 900; color: {t['accent_primary']}; letter-spacing: 1.5px;")

        # Giant Prediction Result Display Box
        self.result_box = QFrame()
        self.result_box.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_main']};
                border: 3px solid {t['accent_primary']};
                border-radius: 16px;
                padding: 20px;
            }}
        """)
        rb_layout = QVBoxLayout(self.result_box)
        rb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_class = QLabel("SCANNING OBJECT...")
        self.lbl_class.setStyleSheet(f"font-size: 34px; font-weight: 900; color: {t['accent_primary']}; letter-spacing: 2px;")
        self.lbl_class.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_conf = QLabel("0.0% CONFIDENCE")
        self.lbl_conf.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {t['accent_secondary']};")
        self.lbl_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_uncertain = QLabel("UNCERTAIN – MOVE OBJECT CLOSER")
        self.lbl_uncertain.setStyleSheet(f"""
            background-color: {t['highlight']};
            color: {t['bg_main']};
            font-weight: 800;
            font-size: 14px;
            padding: 8px 18px;
            border-radius: 14px;
        """)
        self.lbl_uncertain.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_uncertain.setVisible(False)

        rb_layout.addWidget(self.lbl_class)
        rb_layout.addWidget(self.lbl_conf)
        rb_layout.addWidget(self.lbl_uncertain)

        # Giant Confidence Bars
        self.bars_container = QVBoxLayout()
        self.bars_container.setSpacing(12)

        for label in self.classifier.labels:
            guidance = CATEGORY_GUIDANCE.get(label, GENERIC_GUIDANCE)
            icon = guidance.get("icon", "🏷️")
            color = ThemeManager.get_category_color(label)
            bar = ConfidenceProgressBar(category_name=label, icon_str=icon, color_hex=color)
            self.confidence_bars[label] = bar
            self.bars_container.addWidget(bar)

        self.guidance_card = CategoryGuidanceCard()

        r_layout.addWidget(p_title)
        r_layout.addWidget(self.result_box)
        r_layout.addLayout(self.bars_container, 1)
        r_layout.addWidget(self.guidance_card)

        grid_layout.addWidget(self.video_widget, 6)
        grid_layout.addWidget(right_card, 5)

        main_layout.addWidget(header_card)
        main_layout.addLayout(grid_layout, 1)

    def _connect_signals(self):
        self.camera_manager.frame_signal.connect(self.on_frame_received)
        self.btn_exit.clicked.connect(self.close_view)

    @Slot(object, object, tuple, float, bool, str)
    def on_frame_received(self, qimage: QImage, raw_bgr_frame, scan_zone_box: tuple, fps: float, connected: bool, status_msg: str):
        if not self.isVisible():
            return

        self.video_widget.update_frame(qimage)

        if raw_bgr_frame is not None:
            if self.final_decision_locked:
                return

            self.frame_counter += 1
            interval = config_manager.get("inference_interval_frames", 5)
            if self.frame_counter % interval != 0:
                return

            norm_mode = config_manager.get("normalization_mode", "-1_to_1")
            result = self.classifier.predict(raw_bgr_frame, crop_box=scan_zone_box, normalization_mode=norm_mode)

            top_class = result.get("top_class", "")
            top_conf = result.get("top_confidence", 0.0)
            is_confident = result.get("is_confident", True)

            if not is_confident:
                self.candidate_class = None
                self.candidate_count = 0
            elif top_class == self.candidate_class:
                self.candidate_count += 1
            else:
                self.candidate_class = top_class
                self.candidate_count = 1

            if is_confident and self.candidate_count >= 3:
                self.final_decision_locked = True
                play_decision_buzzer()
                self.scan_result_captured.emit(top_class, top_conf)

            smoothed = result.get("smoothed_predictions", {})

            t = ThemeManager.get_theme()
            if is_confident:
                color = ThemeManager.get_category_color(top_class)
                decision_prefix = "FINAL: " if self.final_decision_locked else "CONFIRMING: "
                self.lbl_class.setText(f"{decision_prefix}{top_class.upper()}")
                self.lbl_conf.setText(f"{top_conf:.1f}% CONFIDENCE")
                self.lbl_uncertain.setVisible(False)
                self.lbl_class.setStyleSheet(f"font-size: 34px; font-weight: 900; color: {color}; letter-spacing: 2px;")
                if self.final_decision_locked:
                    self.lbl_uncertain.setText("FINAL DECISION — PRESS R TO SCAN NEXT OBJECT")
                    self.lbl_uncertain.setVisible(True)
            else:
                self.lbl_class.setText("UNCERTAIN")
                self.lbl_conf.setText(f"{top_conf:.1f}% CONFIDENCE")
                self.lbl_uncertain.setVisible(True)
                self.lbl_class.setStyleSheet(f"font-size: 34px; font-weight: 900; color: {t['highlight']}; letter-spacing: 2px;")

            for label, bar in self.confidence_bars.items():
                conf_val = smoothed.get(label, 0.0)
                is_top = (label == top_class) and is_confident
                bar.set_confidence(conf_val, is_top=is_top)
                bar.set_final_decision(self.final_decision_locked and label == top_class)

            if is_confident:
                self.guidance_card.update_guidance(top_class)

    def keyPressEvent(self, event: QKeyEvent):
        """Exit fullscreen when Escape key is pressed."""
        if event.key() == Qt.Key.Key_Escape:
            self.close_view()
        elif event.key() == Qt.Key.Key_R:
            self.reset_decision()
        else:
            super().keyPressEvent(event)

    def reset_decision(self):
        """Clear the exhibition decision so the next object can be classified."""
        self.final_decision_locked = False
        self.candidate_class = None
        self.candidate_count = 0
        self.classifier.history_buffer.clear()
        for bar in self.confidence_bars.values():
            bar.set_final_decision(False)
        self.lbl_class.setText("SCANNING OBJECT...")
        self.lbl_conf.setText("0.0% CONFIDENCE")
        self.lbl_uncertain.setText("UNCERTAIN – MOVE OBJECT CLOSER")
        self.lbl_uncertain.setVisible(False)

    def close_view(self):
        self.close()
        self.closed_signal.emit()
