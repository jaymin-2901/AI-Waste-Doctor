"""
AI Waste Classifier module for AI Waste Doctor.

Loads the trained 3-class waste classifier, performs inference and smoothing,
and adds a conservative ImageNet semantic guard for obvious food / fruit items.

Why the semantic guard exists:
The custom 3-class model can occasionally classify a visually unfamiliar fruit
as Dry Waste. A pretrained MobileNetV2 recognizer is used only as a second
opinion for strongly recognizable food/produce classes. It does not replace the
waste model and it fails open: if the auxiliary model cannot load, normal waste
classification continues unchanged.
"""

import os
import random
from collections import deque
from pathlib import Path

import numpy as np

from ai.preprocessing import preprocess_frame

# Suppress TensorFlow C++ verbose log messages & oneDNN warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


# ImageNet names emitted by keras.applications.mobilenet_v2.decode_predictions().
# These are deliberately conservative. We only use the auxiliary recognizer to
# correct strong "obvious food / produce" cases into Wet Waste.
WET_WASTE_IMAGENET_LABELS = {
    "granny smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "acorn squash",
    "butternut squash",
    "spaghetti squash",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "broccoli",
    "cauliflower",
    "zucchini",
    "cucumber",
    "head cabbage",
    "corn",
    "ear",
    "guacamole",
    "mashed potato",
    "ice cream",
    "pizza",
    "cheeseburger",
    "hotdog",
    "pretzel",
    "bagel",
    "meat loaf",
    "potpie",
    "burrito",
}


class WasteClassifier:
    """
    Intelligent Waste Classification Engine.

    The main prediction comes from model/keras_model.h5.  A pretrained
    ImageNet MobileNetV2 model can act as a conservative semantic guard for
    obvious food/produce objects (for example, an apple) that the 3-class model
    would otherwise call Dry Waste.
    """

    def __init__(
        self,
        model_path=None,
        labels_path=None,
        smoothing_frames=10,
        confidence_threshold=70.0,
    ):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = (
            Path(model_path)
            if model_path
            else self.base_dir / "model" / "keras_model.h5"
        )
        self.labels_path = (
            Path(labels_path)
            if labels_path
            else self.base_dir / "model" / "labels.txt"
        )

        self.labels = []
        self.model = None
        self.demo_mode = False
        self.status_message = "Initializing AI Classifier..."
        self.model_error = ""

        # Auxiliary semantic recognizer.
        self.semantic_model = None
        self.semantic_guard_enabled = os.getenv(
            "AI_WASTE_SEMANTIC_GUARD", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.semantic_status = "disabled"
        self.semantic_error = ""

        # Stability / smoothing parameters.
        self.smoothing_frames = max(1, smoothing_frames)
        self.confidence_threshold = confidence_threshold
        self.history_buffer = deque(maxlen=self.smoothing_frames)

        self.load_labels()
        self.load_model()

        # Load the auxiliary recognizer after the custom model.  This is
        # intentionally non-fatal: classification still works if ImageNet
        # weights cannot be downloaded/loaded on the deployment.
        if self.semantic_guard_enabled and TF_AVAILABLE:
            self.load_semantic_model()

    def set_smoothing_frames(self, frames: int):
        """Update sliding window size for smoothing filter."""
        self.smoothing_frames = max(1, frames)
        self.history_buffer = deque(maxlen=self.smoothing_frames)

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold percentage."""
        self.confidence_threshold = threshold

    def load_labels(self):
        """Load class names from labels.txt."""
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]

                parsed_labels = []
                for line in lines:
                    parts = line.split(" ", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        parsed_labels.append(parts[1].strip())
                    else:
                        parsed_labels.append(line)

                if parsed_labels:
                    self.labels = parsed_labels
                    print(
                        f"[Classifier] Loaded {len(self.labels)} class labels: "
                        f"{self.labels}"
                    )
                else:
                    self.labels = ["Recyclable", "Dry Waste", "Wet Waste"]
            except Exception as error:
                print(f"[Classifier] Error loading labels.txt: {error}")
                self.labels = ["Recyclable", "Dry Waste", "Wet Waste"]
        else:
            print("[Classifier] labels.txt not found. Using default 3 categories.")
            self.labels = ["Recyclable", "Dry Waste", "Wet Waste"]

    def load_model(self):
        """Load the trained Keras waste model."""
        if not TF_AVAILABLE:
            self.demo_mode = True
            self.status_message = "TensorFlow not installed. DEMO MODE ACTIVE."
            print(f"[Classifier] {self.status_message}")
            return

        model_loaded = False
        saved_model_dir = self.model_path.parent / "saved_model"

        if self.model_path.exists():
            try:
                self.model = tf.keras.models.load_model(
                    str(self.model_path), compile=False
                )
                model_loaded = True
                print(
                    "[Classifier] Successfully loaded Keras model from: "
                    f"{self.model_path}"
                )
            except Exception as error:
                self.model_error = str(error)
                print(
                    f"[Classifier] Error loading model file {self.model_path}: "
                    f"{error}"
                )
        elif saved_model_dir.exists():
            try:
                self.model = tf.keras.models.load_model(
                    str(saved_model_dir), compile=False
                )
                model_loaded = True
                print(
                    "[Classifier] Successfully loaded SavedModel from "
                    f"{saved_model_dir}"
                )
            except Exception as error:
                self.model_error = str(error)
                print(
                    "[Classifier] Error loading SavedModel from "
                    f"{saved_model_dir}: {error}"
                )

        if model_loaded:
            self.demo_mode = False
            self.status_message = "AI Model Loaded Successfully."
        else:
            self.demo_mode = True
            self.status_message = "MODEL NOT FOUND - Running in Demo Mode."
            print(f"[Classifier] {self.status_message}")

    def load_semantic_model(self):
        """
        Load a pretrained ImageNet MobileNetV2 recognizer.

        This model is a second opinion only.  Failure here never switches the
        app into demo mode and never prevents the custom waste model from
        serving predictions.
        """
        if not self.semantic_guard_enabled:
            self.semantic_status = "disabled"
            return

        if not TF_AVAILABLE:
            self.semantic_status = "unavailable"
            self.semantic_error = "TensorFlow is not installed."
            return

        try:
            # Keras downloads the standard ImageNet weights on first use and
            # then keeps them in its local model cache.
            self.semantic_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights="imagenet",
                classifier_activation="softmax",
            )
            self.semantic_model.trainable = False
            self.semantic_status = "ready"
            print("[Classifier] Semantic food guard loaded (ImageNet MobileNetV2).")
        except Exception as error:
            self.semantic_model = None
            self.semantic_status = "unavailable"
            self.semantic_error = str(error)
            print(
                "[Classifier] Semantic food guard unavailable; continuing with "
                f"the custom model only: {error}"
            )

    def _wet_label_index(self):
        """Return the output index that represents wet/organic waste."""
        for index, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized == "wet waste" or "organic" in normalized:
                return index
        return None

    @staticmethod
    def _normalize_semantic_label(label: str) -> str:
        return " ".join(label.replace("_", " ").strip().lower().split())

    def _semantic_food_hint(self, batch_img):
        """
        Return an ImageNet food/produce hint or None.

        We inspect the top 5 predictions.  A hint is accepted only when the
        recognizer has enough combined evidence for known food/produce labels.
        """
        if self.semantic_model is None:
            return None

        try:
            predictions = np.asarray(
                self.semantic_model(batch_img, training=False).numpy(),
                dtype=np.float32,
            )
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=5
            )[0]
        except Exception as error:
            # Do not fail the main classifier because of the auxiliary model.
            self.semantic_error = str(error)
            print(f"[Classifier] Semantic guard inference skipped: {error}")
            return None

        matched = []
        top5 = []

        for class_id, class_name, score in decoded:
            normalized_name = self._normalize_semantic_label(class_name)
            score = float(score)
            top5.append(
                {
                    "id": class_id,
                    "label": normalized_name,
                    "confidence": round(score * 100.0, 1),
                }
            )
            if normalized_name in WET_WASTE_IMAGENET_LABELS:
                matched.append((normalized_name, score))

        if not matched:
            return {
                "matched": False,
                "category": None,
                "label": None,
                "confidence": 0.0,
                "aggregate_confidence": 0.0,
                "top5": top5,
            }

        best_label, best_score = max(matched, key=lambda item: item[1])
        aggregate = min(1.0, sum(score for _, score in matched))

        # Conservative acceptance gate:
        # - at least one mapped food/produce label must have 18% probability
        # - mapped food/produce candidates together must reach at least 25%
        # This prevents the semantic model from overriding the waste model on
        # weak, noisy labels.
        accepted = best_score >= 0.18 and aggregate >= 0.25

        return {
            "matched": accepted,
            "category": "Wet Waste" if accepted else None,
            "label": best_label,
            "confidence": round(best_score * 100.0, 1),
            "aggregate_confidence": round(aggregate * 100.0, 1),
            "top5": top5,
        }

    def _apply_semantic_food_guard(self, raw_probs, hint):
        """
        Fuse a trusted food/produce hint into the 3-class waste probabilities.

        Returns (new_probs, applied).
        """
        if not hint or not hint.get("matched"):
            return raw_probs, False

        wet_index = self._wet_label_index()
        if wet_index is None:
            return raw_probs, False

        probs = np.asarray(raw_probs, dtype=np.float32)
        aggregate = float(hint.get("aggregate_confidence", 0.0)) / 100.0

        # Give obvious food/produce a decisive but not absolute wet-waste vote.
        target_wet = min(0.97, max(0.82, 0.72 + 0.35 * aggregate))
        remaining = 1.0 - target_wet

        other_indices = [index for index in range(len(probs)) if index != wet_index]
        other_total = float(sum(probs[index] for index in other_indices))

        if other_total > 0:
            for index in other_indices:
                probs[index] = (probs[index] / other_total) * remaining
        elif other_indices:
            each = remaining / len(other_indices)
            for index in other_indices:
                probs[index] = each

        probs[wet_index] = target_wet
        probs = probs / probs.sum()
        return probs.tolist(), True

    def predict(self, frame, crop_box=None, normalization_mode="-1_to_1"):
        """
        Perform AI classification on an input frame.

        Returns a dictionary containing raw/smoothed class probabilities,
        confidence status, and optional semantic-guard diagnostics.
        """
        num_classes = len(self.labels)
        batch_img = None
        semantic_hint = None
        semantic_applied = False

        if self.demo_mode or self.model is None:
            raw_probs = self._generate_demo_predictions()
        else:
            try:
                batch_img, _ = preprocess_frame(
                    frame,
                    target_size=(224, 224),
                    normalization_mode=normalization_mode,
                    crop_box=crop_box,
                )

                preds = np.asarray(
                    self.model(batch_img, training=False)[0].numpy(),
                    dtype=np.float32,
                )

                if preds.ndim != 1 or not np.all(np.isfinite(preds)):
                    raise ValueError("Model returned invalid prediction values")

                if len(preds) != num_classes:
                    raise ValueError(
                        f"Model output has {len(preds)} classes, but labels.txt "
                        f"has {num_classes}"
                    )

                if np.all(preds >= 0) and np.isclose(
                    np.sum(preds), 1.0, atol=1e-3
                ):
                    raw_probs = preds.tolist()
                else:
                    shifted = preds - np.max(preds)
                    exp_preds = np.exp(shifted)
                    raw_probs = (exp_preds / np.sum(exp_preds)).tolist()

                # Run the semantic guard only when the custom classifier is not
                # already calling the item Wet Waste.  This keeps normal wet
                # predictions fast while correcting obvious food false negatives.
                base_top_index = int(np.argmax(raw_probs))
                base_top_label = self.labels[base_top_index].strip().lower()
                if (
                    batch_img is not None
                    and self.semantic_model is not None
                    and base_top_label != "wet waste"
                    and "organic" not in base_top_label
                ):
                    semantic_hint = self._semantic_food_hint(batch_img)
                    raw_probs, semantic_applied = self._apply_semantic_food_guard(
                        raw_probs, semantic_hint
                    )

            except Exception as error:
                print(
                    f"[Classifier] Inference error: {error}. "
                    "Falling back to demo mode for this prediction."
                )
                raw_probs = self._generate_demo_predictions()

        self.history_buffer.append(raw_probs)
        buffer_array = np.array(self.history_buffer, dtype=np.float32)
        smoothed_probs = np.mean(buffer_array, axis=0)

        raw_dict = {
            label: float(raw_probs[index] * 100.0)
            for index, label in enumerate(self.labels)
        }
        smoothed_dict = {
            label: float(smoothed_probs[index] * 100.0)
            for index, label in enumerate(self.labels)
        }

        top_index = int(np.argmax(smoothed_probs))
        top_class = self.labels[top_index]
        top_confidence = float(smoothed_probs[top_index] * 100.0)
        is_confident = top_confidence >= self.confidence_threshold

        status = "OK" if is_confident else "UNCERTAIN – MOVE OBJECT CLOSER"
        if semantic_applied and semantic_hint:
            status = (
                "OK · SEMANTIC FOOD GUARD: "
                f"{semantic_hint.get('label', 'food')} → Wet Waste"
            )

        return {
            "raw_predictions": raw_dict,
            "smoothed_predictions": smoothed_dict,
            "top_class": top_class,
            "top_confidence": round(top_confidence, 1),
            "is_confident": is_confident,
            "status": status,
            "demo_mode": self.demo_mode,
            "semantic_guard_applied": semantic_applied,
            "semantic_hint": semantic_hint,
        }

    def _generate_demo_predictions(self):
        """Generate a realistic simulated probability distribution in demo mode."""
        num_classes = len(self.labels)

        if not hasattr(self, "_demo_dominant_idx") or random.random() < 0.1:
            self._demo_dominant_idx = random.randint(0, num_classes - 1)

        probs = [random.uniform(0.02, 0.15) for _ in range(num_classes)]
        probs[self._demo_dominant_idx] = random.uniform(0.75, 0.96)

        total = sum(probs)
        return [probability / total for probability in probs]
