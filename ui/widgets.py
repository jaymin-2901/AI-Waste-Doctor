"""
Custom UI Widgets for AI Waste Doctor.
Contains video display canvas, animated confidence bars, metric summary cards,
and category disposal guidance components.
"""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Slot, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QProgressBar, QVBoxLayout, QHBoxLayout,
    QFrame, QGraphicsDropShadowEffect
)

from config import CATEGORY_GUIDANCE, GENERIC_GUIDANCE
from ui.theme import ThemeManager


class ScanZoneVideoWidget(QLabel):
    """Webcam feed video display label with automatic aspect-ratio scaling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {t['bg_main']};
                border: 2px solid {t['border']};
                border-radius: 12px;
            }}
        """)
        self.current_pixmap = None

    def update_frame(self, qimage: QImage):
        """Update video display with new QImage frame."""
        if not qimage.isNull():
            self.current_pixmap = QPixmap.fromImage(qimage)
            self._rescale_and_set()

    def resizeEvent(self, event):
        """Rescale image on widget resize."""
        super().resizeEvent(event)
        self._rescale_and_set()

    def _rescale_and_set(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)


class ConfidenceProgressBar(QWidget):
    """
    Animated progress bar component representing confidence for a single waste category.
    Highlights when selected as top prediction.
    """

    def __init__(self, category_name: str, icon_str: str = "🏷️", color_hex: str = None, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.icon_str = icon_str
        self.color_hex = color_hex or ThemeManager.get_category_color(category_name)
        self.is_highlighted = False
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(220)
        self.blink_timer.timeout.connect(self._blink_indicator)
        self.blink_count = 0

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        t = ThemeManager.get_theme()

        # Header Row: Icon + Category Name + Percentage Label
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel(self.icon_str)
        self.icon_label.setFont(QFont("Segoe UI Emoji", 14))

        self.name_label = QLabel(self.category_name)
        self.name_label.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {t['text_primary']};")

        self.percent_label = QLabel("0.0%")
        self.percent_label.setStyleSheet(f"font-weight: 800; font-size: 13px; color: {t['text_secondary']};")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.decision_indicator = QLabel()
        self.decision_indicator.setFixedSize(16, 16)
        self.decision_indicator.setToolTip("Final decision")
        self.decision_indicator.setVisible(False)
        self.indicator_glow = QGraphicsDropShadowEffect(self.decision_indicator)
        self.indicator_glow.setBlurRadius(14)
        self.indicator_glow.setOffset(0, 0)
        self.indicator_glow.setEnabled(False)
        self.decision_indicator.setGraphicsEffect(self.indicator_glow)

        header_layout.addWidget(self.icon_label)
        header_layout.addWidget(self.name_label, 1)
        header_layout.addWidget(self.decision_indicator)
        header_layout.addWidget(self.percent_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)  # 0.0% to 100.0% precision
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        self._apply_bar_style(highlight=False)

        layout.addLayout(header_layout)
        layout.addWidget(self.progress_bar)

    def set_confidence(self, percentage: float, is_top: bool = False):
        """Update confidence value (0-100) and highlight state."""
        val_int = int(clamp(percentage, 0.0, 100.0) * 10)
        self.progress_bar.setValue(val_int)
        self.percent_label.setText(f"{percentage:.1f}%")

        if is_top != self.is_highlighted:
            self.is_highlighted = is_top
            self._apply_bar_style(highlight=is_top)

    def set_final_decision(self, active: bool):
        """Blink the category-colored indicator when this class is final."""
        if not active:
            self.blink_timer.stop()
            self.indicator_glow.setEnabled(False)
            self.decision_indicator.setVisible(False)
            return

        self.blink_count = 0
        self.decision_indicator.setStyleSheet(
            f"background-color: {self.color_hex}; border: 2px solid {self.color_hex}; border-radius: 8px;"
        )
        self.indicator_glow.setColor(QColor(self.color_hex))
        self.indicator_glow.setEnabled(True)
        self.decision_indicator.setVisible(True)
        self.blink_timer.start()

    def _blink_indicator(self):
        self.blink_count += 1
        self.decision_indicator.setVisible(not self.decision_indicator.isVisible())
        if self.blink_count >= 8:
            self.blink_timer.stop()
            self.decision_indicator.setVisible(True)

    def _apply_bar_style(self, highlight: bool):
        t = ThemeManager.get_theme()
        if highlight:
            color = QColor(self.color_hex)
            row_surface = (
                f"rgba({color.red()}, {color.green()}, {color.blue()}, 38)"
            )
            self.setStyleSheet(
                f"background-color: {row_surface}; "
                f"border: 2px solid {self.color_hex}; "
                "border-radius: 10px;"
            )
            self.name_label.setStyleSheet(f"font-weight: 800; font-size: 14px; color: {self.color_hex};")
            self.percent_label.setStyleSheet(f"font-weight: 900; font-size: 14px; color: {self.color_hex};")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {t['progress_track']};
                    border: 2px solid {self.color_hex};
                    border-radius: 7px;
                }}
                QProgressBar::chunk {{
                    background-color: {self.color_hex};
                    border-radius: 5px;
                }}
            """)
        else:
            self.setStyleSheet("background-color: transparent; border: none;")
            self.name_label.setStyleSheet(f"font-weight: 600; font-size: 13px; color: {t['text_secondary']};")
            self.percent_label.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {t['text_secondary']};")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {t['progress_track']};
                    border: 1px solid {t['border']};
                    border-radius: 7px;
                }}
                QProgressBar::chunk {{
                    background-color: {t['border']};
                    border-radius: 5px;
                }}
            """)


class MetricCard(QFrame):
    """Dashboard summary metric card displaying title, large bold value, and subtle icon."""

    def __init__(self, title: str, value: str = "0", icon: str = "📊", color: str = None, parent=None):
        super().__init__(parent)
        self.setProperty("class", "CardFrame")
        t = ThemeManager.get_theme()
        color = color or t['accent_primary']
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_card']};
                border: 1px solid {t['border']};
                border-radius: 12px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        top_layout = QHBoxLayout()
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(f"color: {t['text_secondary']}; font-weight: 700; font-size: 11px; letter-spacing: 1px;")
        
        self.icon_lbl = QLabel(icon)
        self.icon_lbl.setFont(QFont("Segoe UI Emoji", 16))

        top_layout.addWidget(self.title_lbl, 1)
        top_layout.addWidget(self.icon_lbl)

        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(f"color: {color}; font-weight: 900; font-size: 28px;")

        layout.addLayout(top_layout)
        layout.addWidget(self.val_lbl)

    def set_value(self, value: str):
        self.val_lbl.setText(str(value))


class CategoryGuidanceCard(QFrame):
    """Visual card providing disposal instructions for the predicted waste category."""

    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {t['bg_secondary']};
                border: 1px solid {t['border']};
                border-radius: 12px;
                padding: 12px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)

        self.icon_lbl = QLabel("♻️")
        self.icon_lbl.setFont(QFont("Segoe UI Emoji", 26))

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        title_lbl = QLabel("DISPOSAL GUIDANCE")
        title_lbl.setStyleSheet(f"color: {t['text_secondary']}; font-weight: 800; font-size: 11px; letter-spacing: 1px;")

        self.text_lbl = QLabel("Place object in scan zone for automated disposal advice.")
        self.text_lbl.setStyleSheet(f"color: {t['text_primary']}; font-weight: 700; font-size: 14px;")
        self.text_lbl.setWordWrap(True)

        info_layout.addWidget(title_lbl)
        info_layout.addWidget(self.text_lbl)

        layout.addWidget(self.icon_lbl)
        layout.addLayout(info_layout, 1)

    def update_guidance(self, category_name: str):
        """Update guidance display based on predicted category."""
        info = CATEGORY_GUIDANCE.get(category_name, GENERIC_GUIDANCE)
        self.icon_lbl.setText(info.get("icon", "🏷️"))
        text = info.get("text", f"Place in designated container for {category_name}")
        color = ThemeManager.get_category_color(category_name)

        self.text_lbl.setText(text)
        self.text_lbl.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 14px;")


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))
