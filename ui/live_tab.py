"""
Live Classification Tab for AI Waste Doctor.
Displays live webcam feed with scan zone, real-time AI Prediction panel with animated progress bars,
stability filter indicators, Manual Scan mode controls, and disposal guidance.
"""

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QRadioButton, QButtonGroup, QMessageBox, QScrollArea
)

from config import CATEGORY_GUIDANCE, GENERIC_GUIDANCE, config_manager
from ui.theme import ThemeManager
from ui.widgets import ScanZoneVideoWidget, ConfidenceProgressBar, CategoryGuidanceCard
from ui.feedback import play_decision_buzzer


class LiveClassificationTab(QWidget):
    """Main live camera & AI prediction interface tab."""

    scan_result_captured = Signal(str, float)  # Signal emitted when manual scan is captured (predicted_class, confidence)

    def __init__(self, camera_manager, classifier, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.classifier = classifier

        # Load default scan mode preference ('manual' or 'live')
        default_mode = config_manager.get("default_scan_mode", "manual")
        self.live_mode = (default_mode == "live")
        self.manual_frozen = False
        self.current_frame_bgr = None
        self.current_scan_box = None
        self.confidence_bars = {}

        # Throttling counter to prevent CPU lag
        self.frame_counter = 0
        self.auto_candidate_class = None
        self.auto_candidate_count = 0
        self.auto_locked = False

        self._build_ui()
        self._connect_signals()

        # Set initial radio button state
        if self.live_mode:
            self.radio_live.setChecked(True)
        else:
            self.radio_manual.setChecked(True)
            self.on_mode_changed(2)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # =========================================================================
        # LEFT PANEL – LIVE CAMERA & CONTROLS
        # =========================================================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)
        
        t = ThemeManager.get_theme()

        # Camera Header Bar (Status Badges)
        cam_header_layout = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA FEED")
        cam_title.setObjectName("H3")

        self.cam_status_badge = QLabel("Camera Disconnected")
        self.cam_status_badge.setStyleSheet(f"""
            background-color: {t['error']};
            color: {t['bg_main']};
            font-weight: 700;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 12px;
        """)

        self.demo_badge = QLabel("DEMO MODE ACTIVE")
        self.demo_badge.setStyleSheet(f"""
            background-color: {t['highlight']};
            color: {t['bg_main']};
            font-weight: 800;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 12px;
        """)
        self.demo_badge.setVisible(self.classifier.demo_mode)

        cam_header_layout.addWidget(cam_title)
        cam_header_layout.addStretch()
        cam_header_layout.addWidget(self.demo_badge)
        cam_header_layout.addWidget(self.cam_status_badge)

        # Video Canvas
        self.video_widget = ScanZoneVideoWidget()

        # Mode Selection Row (Live vs Manual)
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Classification Mode:")
        mode_label.setStyleSheet(f"font-weight: 700; color: {t['text_secondary']};")

        self.mode_group = QButtonGroup(self)
        self.radio_manual = QRadioButton("Manual Button Scan")
        self.radio_live = QRadioButton("Live Auto-Detect")
        self.radio_manual.setStyleSheet(f"font-weight: 700; color: {t['accent_primary']};")
        self.radio_live.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")

        self.mode_group.addButton(self.radio_manual, 2)
        self.mode_group.addButton(self.radio_live, 1)

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.radio_manual)
        mode_layout.addWidget(self.radio_live)
        mode_layout.addStretch()

        # Camera Control Buttons
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setObjectName("SecondaryButton")
        self.btn_switch = QPushButton("Switch Camera")
        self.btn_switch.setObjectName("SecondaryButton")

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_switch)

        # Manual Scan Object Big Button
        self.btn_scan = QPushButton("🔍 SCAN OBJECT")
        self.btn_scan.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['accent_primary']};
                font-size: 16px;
                font-weight: 900;
                padding: 16px;
                border-radius: 12px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {t['accent_secondary']};
            }}
        """)

        # Resume Live Scanning Banner (Hidden by default)
        self.manual_status_card = QFrame()
        self.manual_status_card.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_secondary']};
                border: 1px solid {t['accent_primary']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        manual_card_layout = QHBoxLayout(self.manual_status_card)
        self.manual_msg_label = QLabel("Object scanned successfully!")
        self.manual_msg_label.setStyleSheet(f"font-weight: 700; color: {t['accent_primary']}; font-size: 13px;")
        
        self.btn_resume = QPushButton("Resume Live Auto-Detect")
        self.btn_resume.setObjectName("SecondaryButton")
        self.btn_resume.setFixedHeight(30)
        
        manual_card_layout.addWidget(self.manual_msg_label, 1)
        manual_card_layout.addWidget(self.btn_resume)
        self.manual_status_card.setVisible(False)

        left_panel.addLayout(cam_header_layout)
        left_panel.addWidget(self.video_widget, 1)
        left_panel.addLayout(mode_layout)
        left_panel.addLayout(btn_layout)
        left_panel.addWidget(self.btn_scan)
        left_panel.addWidget(self.manual_status_card)

        # =========================================================================
        # RIGHT PANEL – AI PREDICTION PANEL
        # =========================================================================
        right_card = QFrame()
        right_card.setProperty("class", "CardFrame")

        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(14)

        # Title
        panel_title = QLabel("AI PREDICTION PANEL")
        panel_title.setObjectName("H3")

        # Large Main Result Display Box
        self.result_box = QFrame()
        self.result_box.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_card']};
                border: 2px solid {t['accent_primary']};
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        res_layout = QVBoxLayout(self.result_box)
        res_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.res_class_lbl = QLabel("READY FOR SCAN")
        self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['text_primary']}; letter-spacing: 1px;")
        self.res_class_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.res_conf_lbl = QLabel("Place object in scan zone and press 'SCAN OBJECT'")
        self.res_conf_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {t['text_secondary']};")
        self.res_conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.uncertain_badge = QLabel("UNCERTAIN – MOVE OBJECT CLOSER")
        self.uncertain_badge.setStyleSheet(f"""
            background-color: {t['highlight']};
            color: {t['bg_main']};
            font-weight: 800;
            font-size: 12px;
            padding: 6px 14px;
            border-radius: 12px;
        """)
        self.uncertain_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.uncertain_badge.setVisible(False)

        # Box/Bin indicator
        self.box_indicator = QLabel("")
        self.box_indicator.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {t['accent_primary']}; padding: 8px;")
        self.box_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.box_indicator.setVisible(False)

        res_layout.addWidget(self.res_class_lbl)
        res_layout.addWidget(self.res_conf_lbl)
        res_layout.addWidget(self.uncertain_badge)
        res_layout.addWidget(self.box_indicator)

        # Confidence Bars Section
        bars_title = QLabel("CLASS CONFIDENCE PROBABILITIES")
        bars_title.setStyleSheet(f"font-weight: 800; font-size: 11px; color: {t['text_secondary']}; letter-spacing: 1px;")

        self.bars_container = QVBoxLayout()
        self.bars_container.setSpacing(8)

        # Build initial progress bars for loaded classifier labels
        self.rebuild_confidence_bars()

        # Category Guidance Component
        self.guidance_card = CategoryGuidanceCard()

        right_layout.addWidget(panel_title)
        right_layout.addWidget(self.result_box)
        right_layout.addWidget(bars_title)
        right_layout.addLayout(self.bars_container, 1)
        right_layout.addWidget(self.guidance_card)

        # Combine Left & Right
        main_layout.addLayout(left_panel, 6)
        main_layout.addWidget(right_card, 5)

    def rebuild_confidence_bars(self):
        """Reconstruct confidence bar widgets dynamically based on classifier labels."""
        # Clear existing bars
        while self.bars_container.count():
            item = self.bars_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.confidence_bars.clear()

        for label in self.classifier.labels:
            guidance = CATEGORY_GUIDANCE.get(label, GENERIC_GUIDANCE)
            icon = guidance.get("icon", "🏷️")
            color = ThemeManager.get_category_color(label)

            bar_widget = ConfidenceProgressBar(category_name=label, icon_str=icon, color_hex=color)
            self.confidence_bars[label] = bar_widget
            self.bars_container.addWidget(bar_widget)

    def _connect_signals(self):
        self.camera_manager.frame_signal.connect(self.on_frame_received)

        self.btn_start.clicked.connect(self.camera_manager.start_camera)
        self.btn_stop.clicked.connect(self.camera_manager.stop_camera)
        self.btn_switch.clicked.connect(self.on_switch_camera)
        self.btn_scan.clicked.connect(self.on_manual_scan_clicked)
        self.btn_resume.clicked.connect(self.on_resume_live_scan)

        self.mode_group.idClicked.connect(self.on_mode_changed)

    @Slot(object, object, tuple, float, bool, str)
    def on_frame_received(self, qimage: QImage, raw_bgr_frame, scan_zone_box: tuple, fps: float, connected: bool, status_msg: str):
        """Slot executed whenever a new frame arrives from the camera worker."""
        # Update camera status badge
        t = ThemeManager.get_theme()
        if connected:
            self.cam_status_badge.setText("Camera Connected")
            self.cam_status_badge.setStyleSheet(f"""
                background-color: {t['bg_secondary']};
                color: {t['accent_primary']};
                font-weight: 700;
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 12px;
            """)
        else:
            self.cam_status_badge.setText(status_msg)
            self.cam_status_badge.setStyleSheet(f"""
                background-color: {t['error']};
                color: {t['bg_main']};
                font-weight: 700;
                font-size: 11px;
                padding: 4px 10px;
                border-radius: 12px;
            """)

        self.demo_badge.setVisible(self.classifier.demo_mode)

        # Store latest frame & crop box
        self.current_frame_bgr = raw_bgr_frame
        self.current_scan_box = scan_zone_box

        # Render video feed on screen
        if not self.manual_frozen:
            self.video_widget.update_frame(qimage)

            # Perform AI Prediction ONLY if in continuous Live Mode and interval matched
            if self.live_mode and raw_bgr_frame is not None:
                if self.auto_locked:
                    return

                self.frame_counter += 1
                interval = config_manager.get("inference_interval_frames", 5)
                if self.frame_counter % interval == 0:
                    norm_mode = config_manager.get("normalization_mode", "-1_to_1")
                    result = self.classifier.predict(raw_bgr_frame, crop_box=scan_zone_box, normalization_mode=norm_mode)
                    self._process_auto_detection(result)

    def _process_auto_detection(self, result: dict):
        """Confirm one stable class, then keep that final decision on screen."""
        top_class = result.get("top_class", "")
        is_confident = result.get("is_confident", False)

        if not is_confident:
            self.auto_candidate_class = None
            self.auto_candidate_count = 0
            self.update_prediction_ui(result)
            return

        if top_class == self.auto_candidate_class:
            self.auto_candidate_count += 1
        else:
            self.auto_candidate_class = top_class
            self.auto_candidate_count = 1

        required_frames = 3
        if self.auto_candidate_count >= required_frames:
            self.auto_locked = True
            play_decision_buzzer()
            self.update_prediction_ui(result, final_decision=True)
            self.manual_msg_label.setText(
                f"Final Decision: {top_class} ({result.get('top_confidence', 0.0):.1f}% Confidence)"
            )
            self.manual_status_card.setVisible(True)
        else:
            self.update_prediction_ui(result)

    def update_prediction_ui(self, result: dict, final_decision: bool = False):
        """Update the AI Prediction Panel controls with prediction result dictionary."""
        top_class = result.get("top_class", "Organic Waste")
        top_conf = result.get("top_confidence", 0.0)
        is_confident = result.get("is_confident", True)
        smoothed = result.get("smoothed_predictions", {})

        t = ThemeManager.get_theme()
        if is_confident:
            guidance = CATEGORY_GUIDANCE.get(top_class, GENERIC_GUIDANCE)
            color = ThemeManager.get_category_color(top_class)
            box_color = guidance.get("box_color", "")

            self.res_class_lbl.setText(top_class.upper())
            self.res_conf_lbl.setText(f"{top_conf:.1f}% CONFIDENCE")
            self.uncertain_badge.setVisible(False)
            self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {color}; letter-spacing: 1px;")

            self.box_indicator.setText(f"📦 Put this in the {box_color.upper()} BOX")
            self.box_indicator.setStyleSheet(f"""
                font-size: 16px; font-weight: 800; color: {color}; padding: 8px;
                background-color: {t['bg_secondary']}; border-radius: 8px;
            """)
            self.box_indicator.setVisible(True)
        else:
            self.res_class_lbl.setText("UNCERTAIN")
            self.res_conf_lbl.setText(f"{top_conf:.1f}% CONFIDENCE")
            self.uncertain_badge.setVisible(True)
            self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['highlight']}; letter-spacing: 1px;")
            self.box_indicator.setVisible(False)

        # Update each class progress bar
        for label, bar in self.confidence_bars.items():
            conf_val = smoothed.get(label, 0.0)
            is_top = (label == top_class) and is_confident
            bar.set_confidence(conf_val, is_top=is_top)
            bar.set_final_decision(final_decision and label == top_class)

        # Update disposal guidance card
        if is_confident:
            self.guidance_card.update_guidance(top_class)
        else:
            self.guidance_card.update_guidance("Uncertain")

        # Emit capture signal for scientific experiment auto-filling
        self.scan_result_captured.emit(top_class if is_confident else "", top_conf)

    @Slot()
    def on_manual_scan_clicked(self):
        """Capture single frame and lock prediction in Manual Scan Mode."""
        if self.current_frame_bgr is None:
            QMessageBox.warning(self, "No Video Frame", "Cannot scan: No valid camera frame available.")
            return

        norm_mode = config_manager.get("normalization_mode", "-1_to_1")
        result = self.classifier.predict(self.current_frame_bgr, crop_box=self.current_scan_box, normalization_mode=norm_mode)
        
        self.manual_frozen = True
        play_decision_buzzer()
        self.update_prediction_ui(result, final_decision=True)

        top_c = result["top_class"]
        top_conf = result["top_confidence"]

        self.manual_msg_label.setText(f"Object Scanned: {top_c} ({top_conf:.1f}% Confidence)")
        self.manual_status_card.setVisible(True)

    @Slot()
    def on_resume_live_scan(self):
        """Resume continuous live video scanning."""
        self.manual_frozen = False
        self.auto_locked = False
        self.auto_candidate_class = None
        self.auto_candidate_count = 0
        self.classifier.history_buffer.clear()
        self.manual_status_card.setVisible(False)
        self.radio_live.setChecked(True)
        self.live_mode = True

    @Slot(int)
    def on_mode_changed(self, mode_id: int):
        """Handle Live vs Manual radio button toggle."""
        t = ThemeManager.get_theme()
        if mode_id == 1:
            # Live Auto-Detect Mode
            self.live_mode = True
            self.manual_frozen = False
            self.auto_locked = False
            self.auto_candidate_class = None
            self.auto_candidate_count = 0
            self.classifier.history_buffer.clear()
            self.manual_status_card.setVisible(False)
            self.radio_live.setStyleSheet(f"font-weight: 700; color: {t['accent_primary']};")
            self.radio_manual.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        else:
            # Manual Button Scan Mode
            self.live_mode = False
            self.auto_locked = False
            self.auto_candidate_class = None
            self.auto_candidate_count = 0
            self.classifier.history_buffer.clear()
            self.radio_manual.setStyleSheet(f"font-weight: 700; color: {t['accent_primary']};")
            self.radio_live.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
            if not self.manual_frozen:
                self.res_class_lbl.setText("READY FOR SCAN")
                self.res_conf_lbl.setText("Place object in scan zone and press 'SCAN OBJECT'")
                self.uncertain_badge.setVisible(False)

    @Slot()
    def on_switch_camera(self):
        """Cycle through camera indices 0, 1, 2."""
        curr = self.camera_manager.camera_index
        next_idx = (curr + 1) % 3
        config_manager.set("camera_index", next_idx)
        self.camera_manager.set_camera_index(next_idx)
