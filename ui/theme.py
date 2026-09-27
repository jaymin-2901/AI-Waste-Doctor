"""
Theme manager and color palettes for AI Waste Doctor.
Supports dynamic switching between Light and Dark modes.
"""

from PySide6.QtGui import QColor

class Theme:
    LIGHT = {
        "name": "light",
        "bg_main": "#F8F7F4",
        "bg_card": "#FFFFFF",
        "bg_secondary": "#EEEDE8",
        "accent_primary": "#4D7C0F",
        "accent_secondary": "#0EA5E9",
        "highlight": "#F59E0B",
        "error": "#DC2626",
        "text_primary": "#15221B",
        "text_secondary": "#5B6B62",
        "border": "#D7E3DC",
        "progress_track": "#E5E7EB",
        
        # Category Colors
        "cat_organic": "#16A34A",
        "cat_paper": "#2563EB",
        "cat_plastic": "#D97706",
        "cat_uncertain": "#7C3AED"
    }

    DARK = {
        "name": "dark",
        "bg_main": "#23262F",
        "bg_card": "#2B2E38",
        "bg_secondary": "#343741",
        "accent_primary": "#B6FF2E",
        "accent_secondary": "#38BDF8",
        "highlight": "#FBBF24",
        "error": "#F87171",
        "text_primary": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "border": "#2B2B2B",
        "progress_track": "#252525",
        
        # Category Colors
        "cat_organic": "#22C55E",
        "cat_paper": "#60A5FA",
        "cat_plastic": "#FBBF24",
        "cat_uncertain": "#A78BFA"
    }

class ThemeManager:
    """Manages the current active theme."""
    _current_theme = Theme.DARK

    @classmethod
    def set_theme(cls, is_dark: bool):
        cls._current_theme = Theme.DARK if is_dark else Theme.LIGHT

    @classmethod
    def get_theme(cls) -> dict:
        return cls._current_theme

    @classmethod
    def get_color(cls, key: str) -> str:
        return cls._current_theme.get(key, cls._current_theme["text_primary"])

    @classmethod
    def get_category_color(cls, category: str) -> str:
        if category in ("Organic Waste", "Wet Waste"):
            return cls.get_color("cat_organic")
        elif category in ("Paper/Cardboard", "Dry Waste"):
            return cls.get_color("cat_paper")
        elif category in ("Plastic/Metal", "Recyclable"):
            return cls.get_color("cat_plastic")
        else:
            return cls.get_color("cat_uncertain")

def get_app_style(is_dark: bool) -> str:
    """Returns the global QSS string for the requested theme."""
    t = Theme.DARK if is_dark else Theme.LIGHT
    
    return f"""
    /* Global Application Style */
    QWidget {{
        background-color: {t['bg_main']};
        color: {t['text_primary']};
        font-family: 'Aptos', 'Segoe UI', sans-serif;
        font-size: 13px;
    }}

    QMainWindow {{
        background-color: {t['bg_main']};
    }}

    QLabel#AppHeaderTitle {{
        color: {t['accent_primary']};
        font-family: 'Aptos Display', 'Aptos', 'Segoe UI', sans-serif;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: 1px;
    }}

    QLabel#AppHeaderSubtitle {{
        color: {t['text_secondary']};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    /* ScrollBar Styling */
    QScrollBar:vertical {{
        border: none;
        background: {t['bg_main']};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {t['border']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {t['accent_primary']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* Navigation Tab Bar */
    QTabWidget::pane {{
        border: 1px solid {t['border']};
        background: {t['bg_main']};
        border-radius: 8px;
    }}
    QTabBar::tab {{
        background: {t['bg_secondary']};
        color: {t['text_secondary']};
        padding: 11px 20px;
        font-weight: 600;
        font-size: 13px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
        border: 1px solid {t['border']};
        border-bottom: none;
    }}
    QTabBar::tab:hover {{
        background: {t['bg_card']};
        color: {t['text_primary']};
    }}
    QTabBar::tab:selected {{
        background: {t['accent_primary']};
        color: {t['bg_main']};
        font-weight: 700;
        border: none;
    }}
    QTabBar::tab:selected:hover {{
        background: {t['accent_primary']};
    }}

    /* Cards & Frames */
    QFrame.CardFrame {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 12px;
    }}

    QFrame.AccentCard {{
        background-color: {t['bg_secondary']};
        border: 1px solid {t['accent_primary']};
        border-radius: 12px;
    }}

    QStatusBar {{
        background-color: {t['bg_card']};
        color: {t['text_secondary']};
        border-top: 1px solid {t['border']};
        font-size: 11px;
        padding: 4px 8px;
    }}

    /* Primary Action Buttons */
    QPushButton {{
        background-color: {t['accent_primary']};
        color: {t['bg_main']};
        font-weight: 700;
        font-size: 13px;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
    }}
    QPushButton:hover {{
        background-color: {t['accent_secondary']};
    }}
    QPushButton:pressed {{
        background-color: {t['accent_primary']};
        padding-top: 11px;
        padding-bottom: 9px;
    }}
    QPushButton:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 2px solid {t['accent_secondary']};
    }}
    QPushButton:disabled {{
        background-color: {t['bg_secondary']};
        color: {t['text_secondary']};
    }}

    /* Secondary Buttons */
    QPushButton#SecondaryButton {{
        background-color: {t['bg_secondary']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
    }}
    QPushButton#SecondaryButton:hover {{
        background-color: {t['border']};
    }}

    QPushButton#ScienceFairBtn {{
        background-color: {t['accent_primary']};
        color: {t['bg_main']};
        border: 2px solid {t['accent_primary']};
        font-weight: 900;
        padding: 10px 16px;
    }}
    QPushButton#ScienceFairBtn:hover {{
        background-color: {t['accent_primary']};
        color: {t['bg_main']};
        border-color: {t['text_primary']};
    }}

    /* Form Controls */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
    }}
    QComboBox:hover, QSpinBox:hover, QLineEdit:hover {{
        border-color: {t['accent_primary']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QToolTip {{
        background-color: {t['bg_card']};
        color: {t['text_primary']};
        border: 1px solid {t['accent_secondary']};
        padding: 6px 8px;
    }}

    /* CheckBox */
    QCheckBox {{
        color: {t['text_primary']};
        font-size: 13px;
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid {t['border']};
        background-color: {t['bg_card']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {t['accent_primary']};
        border-color: {t['accent_primary']};
    }}

    QRadioButton {{
        color: {t['text_primary']};
        spacing: 8px;
        font-weight: 600;
    }}
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 2px solid {t['border']};
        border-radius: 8px;
        background-color: {t['bg_card']};
    }}
    QRadioButton::indicator:checked {{
        background-color: {t['accent_primary']};
        border-color: {t['accent_primary']};
    }}

    /* Table Widget */
    QTableWidget {{
        background-color: {t['bg_card']};
        alternate-background-color: {t['bg_secondary']};
        gridline-color: {t['border']};
        border: 1px solid {t['border']};
        border-radius: 8px;
    }}
    QHeaderView::section {{
        background-color: {t['bg_secondary']};
        color: {t['accent_primary']};
        font-weight: 700;
        padding: 8px;
        border: none;
        border-bottom: 2px solid {t['accent_primary']};
    }}
    QTableWidget::item {{
        padding: 6px;
    }}
    QTableWidget::item:selected {{
        background-color: {t['accent_primary']};
        color: {t['bg_main']};
    }}
    QTableCornerButton::section {{
        background-color: {t['bg_secondary']};
        border: none;
    }}

    /* Progress Bar */
    QProgressBar {{
        background-color: {t['progress_track']};
        border: none;
        border-radius: 6px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background-color: {t['accent_primary']};
        border-radius: 6px;
    }}
    
    /* Headings */
    QLabel#H1 {{
        font-family: 'Aptos Display', 'Aptos', 'Segoe UI', sans-serif;
        font-size: 22px;
        font-weight: 900;
        color: {t['accent_primary']};
    }}
    QLabel#H2 {{
        font-family: 'Aptos Display', 'Aptos', 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: 800;
        color: {t['text_primary']};
    }}
    QLabel#H3 {{
        font-family: 'Aptos Display', 'Aptos', 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: {t['text_secondary']};
    }}
    """
