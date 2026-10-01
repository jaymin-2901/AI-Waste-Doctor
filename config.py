"""
Configuration & Settings Manager for AI Waste Doctor.
Handles persistent JSON settings for camera, smoothing, threshold, and display preferences.
"""

import os
import json
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"

# File Locations
CONFIG_FILE = DATA_DIR / "config.json"
CSV_FILE = DATA_DIR / "results.csv"
LABELS_FILE = MODEL_DIR / "labels.txt"
MODEL_FILE = MODEL_DIR / "keras_model.h5"

# Default Configuration Values
DEFAULT_CONFIG = {
    "camera_index": 0,
    "confidence_threshold": 70.0,
    "smoothing_frames": 10,
    "mirror_webcam": True,
    "show_fps": True,
    "sound_enabled": False,
    "normalization_mode": "-1_to_1",  # MobileNetV2/Teachable Machine: (pixel/127.5)-1
    "demo_mode": False,
    "scan_zone_ratio": 0.6,  # 60% of video height/width
    "inference_interval_frames": 5,  # Run AI inference every N frames (saves 80%+ CPU lag)
    "live_decision_frames": 3,
    "default_scan_mode": "manual",  # "manual" (only scan when button pressed) or "live" (continuous)
    "theme": "dark"  # "dark" or "light"
}

# Disposal Guidance dictionary per category
CATEGORY_GUIDANCE = {
    "Recyclable": {
        "text": "Place in the BLUE RECYCLING BIN — bottles, cans, glass, packaging, and electronics at an approved e-waste center.",
        "icon": "♻️",
        "box_color": "Blue"
    },
    "Dry Waste": {
        "text": "Place in the YELLOW DRY-WASTE BIN — wrappers, styrofoam, and non-recyclable packaging.",
        "icon": "📦",
        "box_color": "Yellow"
    },
    "Wet Waste": {
        "text": "Place in the GREEN COMPOST BIN — food scraps and garden waste.",
        "icon": "🌿",
        "box_color": "Green"
    },
    "Organic Waste": {
        "text": "Place in the GREEN COMPOST BIN — Food scraps, organic materials.",
        "icon": "🌿",
        "box_color": "Green"
    },
    "Paper/Cardboard": {
        "text": "Place in the BLUE RECYCLING BIN — Paper, cardboard.",
        "icon": "📄",
        "box_color": "Blue"
    },
    "Plastic/Metal": {
        "text": "Place in the YELLOW RECYCLING BIN — Plastics, metals, glass.",
        "icon": "♻️",
        "box_color": "Yellow"
    }
}

# Generic fallback guidance for any custom label in labels.txt
GENERIC_GUIDANCE = {
    "text": "Place in designated sorting bin for this material",
    "icon": "🏷️",
    "box_color": "Grey"
}


class ConfigManager:
    """Manages reading and writing application settings to config.json."""
    
    def __init__(self):
        self.ensure_directories()
        self.settings = DEFAULT_CONFIG.copy()
        self.settings = self.load_settings()

    def ensure_directories(self):
        """Ensure data, model, and assets directories exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    def load_settings(self) -> dict:
        """Load settings from JSON file or create defaults if missing."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with default config to ensure all keys exist
                    config = DEFAULT_CONFIG.copy()
                    config.update(data)
                    return config
            except Exception as e:
                print(f"[ConfigManager] Error reading config file: {e}. Using defaults.")
                return DEFAULT_CONFIG.copy()
        else:
            self.save_settings(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

    def save_settings(self, new_settings: dict = None):
        """Save settings dictionary to JSON file."""
        if not hasattr(self, "settings") or self.settings is None:
            self.settings = DEFAULT_CONFIG.copy()
        if new_settings is not None:
            self.settings.update(new_settings)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"[ConfigManager] Error saving config file: {e}")

    def get(self, key: str, default=None):
        """Get a setting value by key."""
        return self.settings.get(key, default)

    def set(self, key: str, value):
        """Set a setting value and persist."""
        self.settings[key] = value
        self.save_settings()


# Global instance
config_manager = ConfigManager()
