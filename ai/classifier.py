"""AI Waste Doctor classification engine.

The trained 3-class waste model remains the primary classifier.  A lightweight
ImageNet MobileNetV2 cross-check is used only when the waste model predicts a
non-wet class.  The cross-check is object-centric and evaluates several nested
center crops so a fruit placed inside a bowl does not get lost in background or
container pixels.
"""

import os
import random
from collections import deque
from pathlib import Path

import numpy as np

from ai.preprocessing import preprocess_frame

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


# Keras / ImageNet output indices.  The produce block is contiguous in the
# canonical ImageNet-1K class ordering used by MobileNetV2.
# 936..957 = head cabbage .. pomegranate.
IMAGENET_WET_CLASS_MAP = {
    924: "guacamole",
    925: "consomme",
    926: "hot pot",
    927: "trifle",
    928: "ice cream",
    929: "ice lolly",
    930: "French loaf",
    931: "bagel",
    932: "pretzel",
    933: "cheeseburger",
    934: "hotdog",
    935: "mashed potato",
    936: "head cabbage",
    937: "broccoli",
    938: "cauliflower",
    939: "zucchini",
    940: "spaghetti squash",
    941: "acorn squash",
    942: "butternut squash",
    943: "cucumber",
    944: "artichoke",
    945: "bell pepper",
    946: "cardoon",
    947: "mushroom",
    948: "Granny Smith apple",
    949: "strawberry",
    950: "orange",
    951: "lemon",
    952: "fig",
    953: "pineapple",
    954: "banana",
    955: "jackfruit",
    956: "custard apple",
    957: "pomegranate",
    959: "carbonara",
    962: "meat loaf",
    963: "pizza",
    964: "potpie",
    965: "burrito",
}

PRODUCE_INDICES = tuple(range(936, 958))
COOKED_FOOD_INDICES = (
    924,
    925,
    926,
    927,
    928,
    929,
    930,
    931,
    932,
    933,
    934,
    935,
    959,
    962,
    963,
    964,
    965,
)


class WasteClassifier:
    """Three-class waste classifier with an object-centric food cross-check."""

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

        self.semantic_model = None
        self.semantic_guard_enabled = os.getenv(
            "AI_WASTE_SEMANTIC_GUARD", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.semantic_status = "disabled"
        self.semantic_error = ""

        self.smoothing_frames = max(1, smoothing_frames)
        self.confidence_threshold = float(confidence_threshold)
        self.history_buffer = deque(maxlen=self.smoothing_frames)

        self.load_labels()
        self.load_model()

        if self.semantic_guard_enabled and TF_AVAILABLE:
            # Lazy-load the ImageNet helper only when a scan actually needs it.
            self.semantic_status = "lazy"

    def set_smoothing_frames(self, frames: int):
        self.smoothing_frames = max(1, int(frames))
        self.history_buffer = deque(maxlen=self.smoothing_frames)

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = float(threshold)

    def load_labels(self):
        if self.labels_path.exists():
            try:
                lines = [
                    line.strip()
                    for line in self.labels_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                parsed = []
                for line in lines:
                    parts = line.split(" ", 1)
                    parsed.append(
                        parts[1].strip()
                        if len(parts) > 1 and parts[0].isdigit()
                        else line
                    )
                self.labels = parsed or ["Recyclable", "Dry Waste", "Wet Waste"]
                print(f"[Classifier] Loaded labels: {self.labels}")
                return
            except Exception as error:
                print(f"[Classifier] Error loading labels: {error}")

        self.labels = ["Recyclable", "Dry Waste", "Wet Waste"]
        print("[Classifier] Using default labels.")

    def load_model(self):
        if not TF_AVAILABLE:
            self.demo_mode = True
            self.status_message = "TensorFlow not installed. DEMO MODE ACTIVE."
            return

        saved_model_dir = self.model_path.parent / "saved_model"
        candidates = [self.model_path, saved_model_dir]

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                self.model = tf.keras.models.load_model(str(candidate), compile=False)
                self.demo_mode = False
                self.status_message = "AI Model Loaded Successfully."
                print(f"[Classifier] Loaded waste model from {candidate}")
                return
            except Exception as error:
                self.model_error = str(error)
                print(f"[Classifier] Failed to load {candidate}: {error}")

        self.demo_mode = True
        self.status_message = "MODEL NOT FOUND - Running in Demo Mode."

    def load_semantic_model(self):
        """Load a small pretrained ImageNet recognizer used as a second opinion."""
        if not self.semantic_guard_enabled:
            self.semantic_status = "disabled"
            return
        if not TF_AVAILABLE:
            self.semantic_status = "unavailable"
            self.semantic_error = "TensorFlow is not installed."
            return

        try:
            self.semantic_model = tf.keras.applications.MobileNetV2(
                input_shape=(160, 160, 3),
                alpha=0.35,
                include_top=True,
                weights="imagenet",
                classifier_activation="softmax",
            )
            self.semantic_model.trainable = False
            self.semantic_status = "ready"
            print("[Classifier] Object-centric ImageNet food guard is ready.")
        except Exception as error:
            self.semantic_model = None
            self.semantic_status = "unavailable"
            self.semantic_error = str(error)
            print(f"[Classifier] Food guard unavailable: {error}")

    def _wet_label_index(self):
        for index, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized == "wet waste" or "organic" in normalized:
                return index
        return None

    @staticmethod
    def _probability_row(values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Model returned invalid prediction values")

        if np.all(values >= 0) and np.isclose(float(values.sum()), 1.0, atol=1e-3):
            return values

        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    @staticmethod
    def _scaled_crop_box(frame, crop_box, scale):
        """Scale a crop around its own center while remaining inside the frame."""
        height, width = frame.shape[:2]

        if crop_box is None:
            base_x, base_y, base_w, base_h = 0, 0, width, height
        else:
            base_x, base_y, base_w, base_h = crop_box

        center_x = base_x + (base_w / 2.0)
        center_y = base_y + (base_h / 2.0)
        side = int(max(32, min(base_w, base_h) * float(scale)))
        side = min(side, width, height)

        x = int(round(center_x - side / 2.0))
        y = int(round(center_y - side / 2.0))
        x = max(0, min(x, width - side))
        y = max(0, min(y, height - side))
        return (x, y, side, side)

    def _semantic_food_hint(self, frame, crop_box, base_probs):
        """Evaluate food/produce evidence on several nested object-centric crops."""
        if self.semantic_model is None:
            return None

        try:
            view_scales = (0.68, 0.84, 1.0)
            batches = []
            boxes = []
            for scale in view_scales:
                box = self._scaled_crop_box(frame, crop_box, scale)
                batch, _ = preprocess_frame(
                    frame,
                    target_size=(160, 160),
                    normalization_mode="-1_to_1",
                    crop_box=box,
                )
                batches.append(batch[0])
                boxes.append(box)

            semantic_batch = np.stack(batches, axis=0).astype(np.float32)
            outputs = np.asarray(
                self.semantic_model(semantic_batch, training=False).numpy(),
                dtype=np.float32,
            )
            if outputs.ndim != 2 or outputs.shape[1] < 966:
                raise ValueError(f"Unexpected semantic output shape: {outputs.shape}")
        except Exception as error:
            self.semantic_error = str(error)
            print(f"[Classifier] Semantic inference skipped: {error}")
            return None

        wet_index = self._wet_label_index()
        base_wet = float(base_probs[wet_index]) if wet_index is not None else 0.0

        view_details = []
        best_produce_candidate = None
        best_produce_score = 0.0
        best_food_candidate = None
        best_food_score = 0.0
        max_produce_mass = 0.0
        max_food_mass = 0.0
        produce_supporting_views = 0
        food_supporting_views = 0

        for view_index, row in enumerate(outputs):
            produce_mass = float(np.sum(row[list(PRODUCE_INDICES)]))
            food_mass = float(
                np.sum(row[list(PRODUCE_INDICES)])
                + np.sum(row[list(COOKED_FOOD_INDICES)])
            )

            produce_scores = [
                (index, float(row[index])) for index in PRODUCE_INDICES
            ]
            cooked_scores = [
                (index, float(row[index])) for index in COOKED_FOOD_INDICES
            ]
            local_produce_index, local_produce_score = max(
                produce_scores,
                key=lambda item: item[1],
            )
            local_food_index, local_food_score = max(
                cooked_scores,
                key=lambda item: item[1],
            )

            if local_produce_score > best_produce_score:
                best_produce_score = local_produce_score
                best_produce_candidate = local_produce_index
            if local_food_score > best_food_score:
                best_food_score = local_food_score
                best_food_candidate = local_food_index

            max_produce_mass = max(max_produce_mass, produce_mass)
            max_food_mass = max(max_food_mass, food_mass)
            if produce_mass >= 0.045:
                produce_supporting_views += 1
            if food_mass >= 0.08:
                food_supporting_views += 1

            top_produce = sorted(
                produce_scores,
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            view_details.append(
                {
                    "scale": view_scales[view_index],
                    "crop_box": boxes[view_index],
                    "produce_mass": round(produce_mass * 100.0, 1),
                    "food_mass": round(food_mass * 100.0, 1),
                    "top_produce": [
                        {
                            "label": IMAGENET_WET_CLASS_MAP[index],
                            "confidence": round(score * 100.0, 1),
                        }
                        for index, score in top_produce
                    ],
                }
            )

        produce_match = bool(
            best_produce_candidate is not None
            and (
                best_produce_score >= 0.060
                or (
                    best_produce_score >= 0.015
                    and max_produce_mass >= 0.080
                    and base_wet >= 0.08
                )
                or (
                    best_produce_score >= 0.010
                    and max_produce_mass >= 0.050
                    and produce_supporting_views >= 2
                    and base_wet >= 0.10
                )
            )
        )

        # Cooked-food recognition is intentionally stricter than produce so a
        # printed pizza/burger image on packaging does not easily become Wet Waste.
        cooked_food_match = bool(
            best_food_candidate is not None
            and (
                best_food_score >= 0.14
                or (
                    best_food_score >= 0.065
                    and max_food_mass >= 0.16
                    and food_supporting_views >= 2
                    and base_wet >= 0.14
                )
            )
        )

        best_candidate = (
            best_produce_candidate if produce_match else best_food_candidate
        )
        best_candidate_score = (
            best_produce_score if produce_match else best_food_score
        )
        best_label = (
            IMAGENET_WET_CLASS_MAP.get(best_candidate)
            if best_candidate is not None
            else None
        )

        matched = produce_match or cooked_food_match
        return {
            "matched": matched,
            "category": "Wet Waste" if matched else None,
            "label": best_label,
            "confidence": round(best_candidate_score * 100.0, 1),
            "produce_mass": round(max_produce_mass * 100.0, 1),
            "food_mass": round(max_food_mass * 100.0, 1),
            "supporting_views": produce_supporting_views if produce_match else food_supporting_views,
            "base_wet_confidence": round(base_wet * 100.0, 1),
            "views": view_details,
        }

    def _apply_semantic_food_guard(self, raw_probs, hint):
        if not hint or not hint.get("matched"):
            return raw_probs, False

        wet_index = self._wet_label_index()
        if wet_index is None:
            return raw_probs, False

        probs = np.asarray(raw_probs, dtype=np.float32)
        evidence = max(
            float(hint.get("confidence", 0.0)) / 100.0,
            float(hint.get("produce_mass", 0.0)) / 100.0,
        )
        base_wet = float(hint.get("base_wet_confidence", 0.0)) / 100.0

        target_wet = min(0.94, max(0.78, 0.72 + 0.45 * evidence + 0.25 * base_wet))
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
        num_classes = len(self.labels)
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
                model_output = np.asarray(
                    self.model(batch_img, training=False)[0].numpy(),
                    dtype=np.float32,
                )
                if model_output.size != num_classes:
                    raise ValueError(
                        f"Model output has {model_output.size} classes, but labels.txt has {num_classes}"
                    )
                raw_probs = self._probability_row(model_output).tolist()

                base_top_index = int(np.argmax(raw_probs))
                base_top_label = self.labels[base_top_index].strip().lower()
                if (
                    self.semantic_guard_enabled
                    and base_top_label != "wet waste"
                    and "organic" not in base_top_label
                ):
                    if self.semantic_model is None:
                        self.load_semantic_model()
                    semantic_hint = self._semantic_food_hint(
                        frame,
                        crop_box,
                        raw_probs,
                    )
                    raw_probs, semantic_applied = self._apply_semantic_food_guard(
                        raw_probs,
                        semantic_hint,
                    )
            except Exception as error:
                self.model_error = str(error)
                print(f"[Classifier] Inference error: {error}")
                raise RuntimeError(f"Primary classifier inference failed: {error}") from error

        self.history_buffer.append(raw_probs)
        smoothed_probs = np.mean(
            np.asarray(self.history_buffer, dtype=np.float32),
            axis=0,
        )

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
                "OK · FOOD CROSS-CHECK: "
                f"{semantic_hint.get('label') or 'food/produce'} → Wet Waste"
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
            "semantic_guard_status": self.semantic_status,
        }

    def _generate_demo_predictions(self):
        num_classes = len(self.labels)
        if not hasattr(self, "_demo_dominant_idx") or random.random() < 0.1:
            self._demo_dominant_idx = random.randint(0, num_classes - 1)

        probs = [random.uniform(0.02, 0.15) for _ in range(num_classes)]
        probs[self._demo_dominant_idx] = random.uniform(0.75, 0.96)
        total = sum(probs)
        return [probability / total for probability in probs]
