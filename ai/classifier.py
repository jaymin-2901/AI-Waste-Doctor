"""
AI Waste Classifier module for AI Waste Doctor.
Handles Keras model loading, labels parsing, inference, prediction smoothing, and DEMO MODE.
"""

import os
import random
import numpy as np
from collections import deque
from pathlib import Path
from ai.preprocessing import preprocess_frame

# Suppress TensorFlow C++ verbose log messages & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


class WasteClassifier:
    """
    Intelligent Waste Classification Engine.
    Loads trained TensorFlow/Keras models and labels, performs prediction,
    and applies sliding-window prediction smoothing.
    """

    def __init__(self, model_path=None, labels_path=None, smoothing_frames=10, confidence_threshold=70.0):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else self.base_dir / "model" / "keras_model.h5"
        self.labels_path = Path(labels_path) if labels_path else self.base_dir / "model" / "labels.txt"
        
        self.labels = []
        self.model = None
        self.demo_mode = False
        self.status_message = "Initializing AI Classifier..."
        self.model_error = ""
        
        # Stability / Smoothing Parameters
        self.smoothing_frames = max(1, smoothing_frames)
        self.confidence_threshold = confidence_threshold
        self.history_buffer = deque(maxlen=self.smoothing_frames)
        
        # Load labels and model
        self.load_labels()
        self.load_model()

    def set_smoothing_frames(self, frames: int):
        """Update sliding window size for smoothing filter."""
        self.smoothing_frames = max(1, frames)
        self.history_buffer = deque(maxlen=self.smoothing_frames)

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold percentage."""
        self.confidence_threshold = threshold

    def load_labels(self):
        """Load class names from labels.txt file."""
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                parsed_labels = []
                for line in lines:
                    # Strip numerical prefix if present (e.g. "0 Organic Waste" -> "Organic Waste")
                    parts = line.split(" ", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        parsed_labels.append(parts[1].strip())
                    else:
                        parsed_labels.append(line)
                
                if parsed_labels:
                    self.labels = parsed_labels
                    print(f"[Classifier] Loaded {len(self.labels)} class labels: {self.labels}")
                else:
                    self.labels = ["Organic Waste", "Paper/Cardboard", "Plastic/Metal"]
            except Exception as e:
                print(f"[Classifier] Error loading labels.txt: {e}")
                self.labels = ["Organic Waste", "Paper/Cardboard", "Plastic/Metal"]
        else:
            print("[Classifier] labels.txt not found. Using default 3 categories.")
            self.labels = ["Organic Waste", "Paper/Cardboard", "Plastic/Metal"]

    def load_model(self):
        """Load Keras model (.h5 or SavedModel directory). Fallback to DEMO MODE if absent."""
        if not TF_AVAILABLE:
            self.demo_mode = True
            self.status_message = "TensorFlow not installed. DEMO MODE ACTIVE."
            print(f"[Classifier] {self.status_message}")
            return

        model_loaded = False
        saved_model_dir = self.model_path.parent / "saved_model"

        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                model_loaded = True
                print(f"[Classifier] Successfully loaded Keras model from: {self.model_path}")
            except Exception as e:
                self.model_error = str(e)
                print(f"[Classifier] Error loading model file {self.model_path}: {e}")

        elif saved_model_dir.exists():
            try:
                self.model = tf.keras.models.load_model(str(saved_model_dir), compile=False)
                model_loaded = True
                print(f"[Classifier] Successfully loaded SavedModel from: {saved_model_dir}")
            except Exception as e:
                self.model_error = str(e)
                print(f"[Classifier] Error loading SavedModel from {saved_model_dir}: {e}")

        if model_loaded:
            self.demo_mode = False
            self.status_message = "AI Model Loaded Successfully."
        else:
            self.demo_mode = True
            self.status_message = "MODEL NOT FOUND - Running in Demo Mode."
            print(f"[Classifier] {self.status_message}")

    def predict(self, frame, crop_box=None, normalization_mode="-1_to_1"):
        """
        Perform AI classification on an input frame.
        
        Returns:
        --------
        dict containing:
            "raw_predictions": dict {class_name: confidence_float_0_to_100}
            "smoothed_predictions": dict {class_name: confidence_float_0_to_100}
            "top_class": str
            "top_confidence": float (0-100)
            "is_confident": bool
            "status": str ("OK" or "UNCERTAIN – MOVE OBJECT CLOSER")
            "demo_mode": bool
        """
        num_classes = len(self.labels)

        if self.demo_mode or self.model is None:
            raw_probs = self._generate_demo_predictions()
        else:
            try:
                # Preprocess image frame
                batch_img, _ = preprocess_frame(
                    frame,
                    target_size=(224, 224),
                    normalization_mode=normalization_mode,
                    crop_box=crop_box
                )
                
                # Model inference
                # Direct eager inference avoids Keras' predict data pipeline for
                # one image and keeps memory/thread use low on Render's free tier.
                preds = np.asarray(self.model(batch_img, training=False)[0].numpy(), dtype=np.float32)

                if preds.ndim != 1 or not np.all(np.isfinite(preds)):
                    raise ValueError("Model returned invalid prediction values")
                
                # Check for output dimensionality mismatch with labels
                if len(preds) != num_classes:
                    raise ValueError(
                        f"Model output has {len(preds)} classes, but labels.txt has {num_classes}"
                    )
                
                # Normalize probabilities to sum to 1.0
                # Softmax models already return probabilities. Normalizing again
                # also keeps compatible logits-based exports usable.
                if np.all(preds >= 0) and np.isclose(np.sum(preds), 1.0, atol=1e-3):
                    raw_probs = preds.tolist()
                else:
                    shifted = preds - np.max(preds)
                    exp_preds = np.exp(shifted)
                    raw_probs = (exp_preds / np.sum(exp_preds)).tolist()

            except Exception as e:
                print(f"[Classifier] Inference error: {e}. Falling back to demo mode.")
                raw_probs = self._generate_demo_predictions()

        # Add raw prediction vector to rolling history buffer for smoothing filter
        self.history_buffer.append(raw_probs)

        # Compute averaged probability vector across the latest N frames
        buffer_array = np.array(self.history_buffer)
        smoothed_probs = np.mean(buffer_array, axis=0)

        # Build class-probability dictionaries (in percentage 0-100%)
        raw_dict = {label: float(raw_probs[i] * 100.0) for i, label in enumerate(self.labels)}
        smoothed_dict = {label: float(smoothed_probs[i] * 100.0) for i, label in enumerate(self.labels)}

        # Determine highest confidence class from smoothed probabilities
        top_idx = int(np.argmax(smoothed_probs))
        top_class = self.labels[top_idx]
        top_confidence = float(smoothed_probs[top_idx] * 100.0)

        # Check stability / confidence threshold
        is_confident = top_confidence >= self.confidence_threshold
        status = "OK" if is_confident else "UNCERTAIN – MOVE OBJECT CLOSER"

        return {
            "raw_predictions": raw_dict,
            "smoothed_predictions": smoothed_dict,
            "top_class": top_class,
            "top_confidence": round(top_confidence, 1),
            "is_confident": is_confident,
            "status": status,
            "demo_mode": self.demo_mode
        }

    def _generate_demo_predictions(self):
        """Generate realistic simulated probability distribution in Demo Mode."""
        num_classes = len(self.labels)
        # Randomly favor one class to make demo realistic
        if not hasattr(self, "_demo_dominant_idx") or random.random() < 0.1:
            self._demo_dominant_idx = random.randint(0, num_classes - 1)
        
        probs = [random.uniform(0.02, 0.15) for _ in range(num_classes)]
        probs[self._demo_dominant_idx] = random.uniform(0.75, 0.96)
        
        total = sum(probs)
        return [p / total for p in probs]
