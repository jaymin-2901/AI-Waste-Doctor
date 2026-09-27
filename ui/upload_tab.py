"""
Upload Image Tab for AI Waste Doctor.
Allows the user to browse and load a local image file from storage,
display it on screen, and classify the waste object in the image using the AI model.
"""

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QMessageBox, QSizePolicy
)

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from config import CATEGORY_GUIDANCE, GENERIC_GUIDANCE, config_manager
from ui.theme import ThemeManager
from ui.widgets import ConfidenceProgressBar, CategoryGuidanceCard


class UploadImageTab(QWidget):
    """Tab for uploading a local image file and classifying the waste object in it."""

    # Signal emitted when a scan result is obtained (predicted_class, confidence)
    scan_result_captured = Signal(str, float)

    def __init__(self, classifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self.loaded_image_bgr = None  # Stored as OpenCV BGR numpy array
        self.confidence_bars = {}

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # =========================================================================
        # LEFT PANEL – IMAGE DISPLAY & UPLOAD CONTROLS
        # =========================================================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        t = ThemeManager.get_theme()

        # Header
        header_layout = QHBoxLayout()
        header_title = QLabel("📁 UPLOAD IMAGE FROM DEVICE")
        header_title.setObjectName("H3")

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

        header_layout.addWidget(header_title)
        header_layout.addStretch()
        header_layout.addWidget(self.demo_badge)

        # Image Display Area
        self.image_display = QLabel("No image loaded.\nClick 'Browse Image' to select a waste object photo.")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMinimumSize(480, 360)
        self.image_display.setStyleSheet(f"""
            QLabel {{
                background-color: {t['bg_main']};
                border: 2px dashed {t['border']};
                border-radius: 12px;
                color: {t['text_secondary']};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        self.current_pixmap = None

        # Loaded file path info
        self.lbl_filepath = QLabel("No file selected")
        self.lbl_filepath.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px; font-weight: 600;")

        # Action Buttons Row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_browse = QPushButton("📂 BROWSE IMAGE")
        self.btn_browse.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['accent_secondary']};
                font-size: 14px;
                font-weight: 800;
                padding: 14px 20px;
                border-radius: 10px;
                color: #ffffff;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background-color: {t['accent_primary']};
            }}
        """)

        self.btn_classify = QPushButton("🔍 CLASSIFY OBJECT")
        self.btn_classify.setEnabled(False)
        self.btn_classify.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['accent_primary']};
                font-size: 14px;
                font-weight: 900;
                padding: 14px 20px;
                border-radius: 10px;
                color: #ffffff;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {t['accent_secondary']};
            }}
            QPushButton:disabled {{
                background-color: {t['bg_secondary']};
                color: {t['text_secondary']};
            }}
        """)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("SecondaryButton")

        btn_layout.addWidget(self.btn_browse, 2)
        btn_layout.addWidget(self.btn_classify, 2)
        btn_layout.addWidget(self.btn_clear, 1)

        # Scan result info card (appears after classification)
        self.scan_result_card = QFrame()
        self.scan_result_card.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_secondary']};
                border: 1px solid {t['accent_primary']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        result_card_layout = QHBoxLayout(self.scan_result_card)
        self.scan_result_msg = QLabel("Image classified successfully!")
        self.scan_result_msg.setStyleSheet(f"font-weight: 700; color: {t['accent_primary']}; font-size: 13px;")
        self.scan_result_msg.setWordWrap(True)
        result_card_layout.addWidget(self.scan_result_msg)
        self.scan_result_card.setVisible(False)

        left_panel.addLayout(header_layout)
        left_panel.addWidget(self.image_display, 1)
        left_panel.addWidget(self.lbl_filepath)
        left_panel.addLayout(btn_layout)
        left_panel.addWidget(self.scan_result_card)

        # =========================================================================
        # RIGHT PANEL – AI PREDICTION PANEL
        # =========================================================================
        right_card = QFrame()
        right_card.setProperty("class", "CardFrame")

        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(14)

        panel_title = QLabel("AI PREDICTION PANEL")
        panel_title.setObjectName("H3")

        # Result display box
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

        self.res_class_lbl = QLabel("UPLOAD AN IMAGE")
        self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['text_primary']}; letter-spacing: 1px;")
        self.res_class_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.res_conf_lbl = QLabel("Browse a waste object photo and click 'Classify Object'")
        self.res_conf_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {t['text_secondary']};")
        self.res_conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.uncertain_badge = QLabel("UNCERTAIN – IMAGE MAY NOT CONTAIN A RECOGNIZED OBJECT")
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
        self.box_indicator.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 800;
            color: {t['accent_primary']};
            padding: 8px;
        """)
        self.box_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.box_indicator.setVisible(False)

        res_layout.addWidget(self.res_class_lbl)
        res_layout.addWidget(self.res_conf_lbl)
        res_layout.addWidget(self.uncertain_badge)
        res_layout.addWidget(self.box_indicator)

        # Confidence bars section
        bars_title = QLabel("CLASS CONFIDENCE PROBABILITIES")
        bars_title.setStyleSheet(f"font-weight: 800; font-size: 11px; color: {t['text_secondary']}; letter-spacing: 1px;")

        self.bars_container = QVBoxLayout()
        self.bars_container.setSpacing(8)
        self._rebuild_confidence_bars()

        # Guidance card
        self.guidance_card = CategoryGuidanceCard()

        right_layout.addWidget(panel_title)
        right_layout.addWidget(self.result_box)
        right_layout.addWidget(bars_title)
        right_layout.addLayout(self.bars_container, 1)
        right_layout.addWidget(self.guidance_card)

        # Combine panels
        main_layout.addLayout(left_panel, 6)
        main_layout.addWidget(right_card, 5)

    def _rebuild_confidence_bars(self):
        """Rebuild confidence bar widgets from classifier labels."""
        while self.bars_container.count():
            item = self.bars_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.confidence_bars.clear()

        for label in self.classifier.labels:
            guidance = CATEGORY_GUIDANCE.get(label, GENERIC_GUIDANCE)
            icon = guidance.get("icon", "🏷️")
            color = ThemeManager.get_category_color(label)
            bar = ConfidenceProgressBar(category_name=label, icon_str=icon, color_hex=color)
            self.confidence_bars[label] = bar
            self.bars_container.addWidget(bar)

    def _connect_signals(self):
        self.btn_browse.clicked.connect(self.on_browse_clicked)
        self.btn_classify.clicked.connect(self.on_classify_clicked)
        self.btn_clear.clicked.connect(self.on_clear_clicked)

    @Slot()
    def on_browse_clicked(self):
        """Open file dialog to select an image from local storage."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Waste Object Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*)"
        )

        if not filepath:
            return

        try:
            if CV2_AVAILABLE:
                # Decode from bytes so paths containing non-ASCII characters work on Windows.
                img_bgr = cv2.imdecode(
                    np.fromfile(filepath, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
            else:
                QMessageBox.warning(self, "OpenCV Missing", "OpenCV (cv2) is not installed. Cannot load images.")
                return

            if img_bgr is None:
                QMessageBox.warning(self, "Load Error", f"Could not read image file:\n{filepath}")
                return

            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

            self.loaded_image_bgr = img_bgr
            self.lbl_filepath.setText(f"Loaded: {filepath}")

            # Convert BGR to RGB for display
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)

            # Scale to fit display area while keeping aspect ratio
            scaled = pixmap.scaled(
                self.image_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_display.setPixmap(scaled)
            self.current_pixmap = pixmap

            # Update border style to solid now that an image is loaded
            t = ThemeManager.get_theme()
            self.image_display.setStyleSheet(f"""
                QLabel {{
                    background-color: {t['bg_main']};
                    border: 2px solid {t['border']};
                    border-radius: 12px;
                }}
            """)

            self.btn_classify.setEnabled(True)
            self.scan_result_card.setVisible(False)

            # Reset prediction panel
            self.res_class_lbl.setText("IMAGE LOADED")
            self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['text_primary']}; letter-spacing: 1px;")
            self.res_conf_lbl.setText("Click 'Classify Object' to identify the waste type")
            self.uncertain_badge.setVisible(False)
            self.box_indicator.setVisible(False)

        except Exception as e:
            QMessageBox.warning(self, "Error Loading Image", f"An error occurred:\n{str(e)}")

    @Slot()
    def on_classify_clicked(self):
        """Run AI classification on the loaded image."""
        if self.loaded_image_bgr is None:
            QMessageBox.warning(self, "No Image", "Please browse and load an image first.")
            return

        self.demo_badge.setVisible(self.classifier.demo_mode)

        # An upload is an independent scan; do not blend it with camera frames.
        self.classifier.history_buffer.clear()
        norm_mode = config_manager.get("normalization_mode", "-1_to_1")
        try:
            result = self.classifier.predict(
                self.loaded_image_bgr,
                crop_box=None,
                normalization_mode=norm_mode
            )
        except Exception as error:
            QMessageBox.warning(self, "Classification Error", f"Could not classify this image:\n{error}")
            return

        top_class = result.get("top_class", "")
        top_conf = result.get("top_confidence", 0.0)
        is_confident = result.get("is_confident", True)
        smoothed = result.get("smoothed_predictions", {})

        # Update main result display
        t = ThemeManager.get_theme()
        if is_confident:
            self.res_class_lbl.setText(top_class.upper())
            self.res_conf_lbl.setText(f"{top_conf:.1f}% CONFIDENCE")
            self.uncertain_badge.setVisible(False)

            guidance = CATEGORY_GUIDANCE.get(top_class, GENERIC_GUIDANCE)
            color = ThemeManager.get_category_color(top_class)
            box_color = guidance.get("box_color", "")
            self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {color}; letter-spacing: 1px;")

            self.box_indicator.setText(f"📦 Put this in the {box_color.upper()} BOX")
            self.box_indicator.setStyleSheet(f"""
                font-size: 16px;
                font-weight: 800;
                color: {color};
                padding: 8px;
                background-color: {t['bg_secondary']};
                border-radius: 8px;
            """)
            self.box_indicator.setVisible(True)
        else:
            self.res_class_lbl.setText("UNCERTAIN")
            self.res_conf_lbl.setText(f"{top_conf:.1f}% CONFIDENCE")
            self.uncertain_badge.setVisible(True)
            self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['highlight']}; letter-spacing: 1px;")
            self.box_indicator.setVisible(False)

        # Update confidence bars
        for label, bar in self.confidence_bars.items():
            conf_val = smoothed.get(label, 0.0)
            is_top = (label == top_class) and is_confident
            bar.set_confidence(conf_val, is_top=is_top)

        # Update guidance card
        if is_confident:
            self.guidance_card.update_guidance(top_class)
        else:
            self.guidance_card.update_guidance("Uncertain")

        # Show success result card
        result_name = top_class if is_confident else "Uncertain"
        self.scan_result_msg.setText(f"Classification complete: {result_name} ({top_conf:.1f}% confidence)")
        self.scan_result_card.setVisible(True)

        # Emit signal for experiment tab
        self.scan_result_captured.emit(top_class if is_confident else "", top_conf)

    @Slot()
    def on_clear_clicked(self):
        """Reset the upload tab to initial empty state."""
        self.loaded_image_bgr = None
        self.current_pixmap = None
        self.btn_classify.setEnabled(False)
        self.scan_result_card.setVisible(False)

        t = ThemeManager.get_theme()
        self.image_display.setText("No image loaded.\nClick 'Browse Image' to select a waste object photo.")
        self.image_display.setPixmap(QPixmap())
        self.image_display.setStyleSheet(f"""
            QLabel {{
                background-color: {t['bg_main']};
                border: 2px dashed {t['border']};
                border-radius: 12px;
                color: {t['text_secondary']};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        self.lbl_filepath.setText("No file selected")

        self.res_class_lbl.setText("UPLOAD AN IMAGE")
        self.res_class_lbl.setStyleSheet(f"font-size: 24px; font-weight: 900; color: {t['text_primary']}; letter-spacing: 1px;")
        self.res_conf_lbl.setText("Browse a waste object photo and click 'Classify Object'")
        self.uncertain_badge.setVisible(False)
        self.box_indicator.setVisible(False)

        for bar in self.confidence_bars.values():
            bar.set_confidence(0.0, is_top=False)
