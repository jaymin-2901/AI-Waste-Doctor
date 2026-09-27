"""
How It Works Tab for AI Waste Doctor.
Educational science fair documentation panel explaining machine learning pipeline,
computer vision feature extraction, preprocessing, and model inference to judges and visitors.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea
)
from ui.theme import ThemeManager


class HowItWorksTab(QWidget):
    """Educational flowchart and explanation tab for science fair visitors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)
        
        t = ThemeManager.get_theme()

        # Title
        header_lbl = QLabel("HOW THE AI WASTE DOCTOR WORKS")
        header_lbl.setObjectName("H2")

        # Visual Pipeline Flow Card
        pipeline_card = QFrame()
        pipeline_card.setProperty("class", "CardFrame")
        pipe_layout = QVBoxLayout(pipeline_card)

        pipe_title = QLabel("SYSTEM ARCHITECTURE PIPELINE")
        pipe_title.setObjectName("H3")

        # Pipeline diagram nodes
        diagram_layout = QHBoxLayout()
        diagram_layout.setSpacing(8)

        steps = [
            ("📁", "INPUT", "Upload Image or\nLive Webcam Feed"),
            ("🖼️", "IMAGE PIXELS", "Read Pixels\nBGR Matrices"),
            ("⚙️", "PREPROCESSING", "Resize (224×224)\nNormalize Pixels"),
            ("🧠", "AI MODEL", "Deep CNN\nFeature Extraction"),
            ("📊", "PROBABILITY", "Softmax Scores\nSmoothing Filter"),
            ("📦", "SORTING BOX", "Recyclable / Dry\nWet Waste Bin")
        ]

        for idx, (icon, step_title, desc) in enumerate(steps):
            box = QFrame()
            box.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_card']};
                    border: 1px solid {t['border']};
                    border-radius: 10px;
                    padding: 8px;
                }}
            """)
            box_lay = QVBoxLayout(box)
            box_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

            icon_lbl = QLabel(icon)
            icon_lbl.setFont(QFont("Segoe UI Emoji", 20))
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            title_lbl = QLabel(step_title)
            title_lbl.setStyleSheet(f"font-size: 11px; font-weight: 900; color: {t['accent_primary']};")
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f"font-size: 10px; color: {t['text_secondary']};")
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            box_lay.addWidget(icon_lbl)
            box_lay.addWidget(title_lbl)
            box_lay.addWidget(desc_lbl)

            diagram_layout.addWidget(box, 1)

            if idx < len(steps) - 1:
                arrow = QLabel("➔")
                arrow.setStyleSheet(f"font-size: 16px; font-weight: 900; color: {t['accent_primary']};")
                diagram_layout.addWidget(arrow)

        pipe_layout.addWidget(pipe_title)
        pipe_layout.addLayout(diagram_layout)

        # Educational Explanation Card
        info_card = QFrame()
        info_card.setProperty("class", "CardFrame")
        info_layout = QVBoxLayout(info_card)
        info_layout.setSpacing(14)

        sec1_title = QLabel("💡 Core Concept: Computer Vision & Deep Learning")
        sec1_title.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {t['text_primary']};")

        sec1_text = QLabel(
            "The AI model examines visual features such as shape, colour, edges and texture and calculates "
            "probabilities for each waste category: Recyclable, Dry Waste, and Wet Waste.\n\n"
            "Using a Convolutional Neural Network (CNN), the system analyzes images — either uploaded from "
            "your device or captured live with a webcam. Computers see images as numbers representing "
            "Red, Green, and Blue pixel intensities ranging from 0 to 255."
        )
        sec1_text.setStyleSheet(f"color: {t['text_secondary']}; font-size: 13px; line-height: 1.5;")
        sec1_text.setWordWrap(True)

        sec2_title = QLabel("🔬 Two Detection Modes: Upload & Live Camera")
        sec2_title.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {t['text_primary']};")

        sec2_text = QLabel(
            "Mode 1 — UPLOAD IMAGE: Browse and select a photo of a waste object from your device. "
            "The AI classifies it instantly with no camera lag.\n\n"
            "Mode 2 — LIVE CAMERA: Point your webcam at a waste object and scan in real-time. "
            "The system filters out non-object detections (humans, background noise) and only identifies waste items.\n\n"
            "After classification, the app tells you exactly which cardboard sorting box to place the item into: "
            "Blue Box (Recyclable), Green Box (Dry Waste), or Red Box (Wet Waste)."
        )
        sec2_text.setStyleSheet(f"color: {t['text_secondary']}; font-size: 13px; line-height: 1.5;")
        sec2_text.setWordWrap(True)

        info_layout.addWidget(sec1_title)
        info_layout.addWidget(sec1_text)
        info_layout.addWidget(sec2_title)
        info_layout.addWidget(sec2_text)

        scroll_layout.addWidget(header_lbl)
        scroll_layout.addWidget(pipeline_card)
        scroll_layout.addWidget(info_card)
        scroll.setWidget(scroll_widget)

        main_layout.addWidget(scroll)
