"""AI Waste Doctor classification engine.

The trained three-class waste model is the primary classifier.  Science Fair
mode additionally uses an object-centric ImageNet MobileNetV2 second opinion.
The second opinion is deliberately conservative: it can promote strong food
evidence to Wet Waste and can demote a clearly recognized bottle/plastic/electronic
object away from Wet Waste.  This prevents a single bad 3-class prediction from
being presented as a confident final bin decision.
"""

import os
import random
from collections import deque
from pathlib import Path

import cv2
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


# MobileNetV2/ImageNet-1K indices.  These are used only as a second opinion;
# the trained waste model still supplies the application probabilities.
IMAGENET_CLASS_MAP = {
    # food / produce
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
    # strong non-wet object signals
    487: "cellular telephone",
    508: "computer keyboard",
    620: "laptop",
    664: "monitor",
    675: "computer mouse",
    725: "plastic bag",
    734: "pop bottle",
    741: "remote control",
    898: "water bottle",
}

PRODUCE_INDICES = tuple(range(936, 958))
COOKED_FOOD_INDICES = (924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 959, 962, 963, 964, 965)
RECYCLABLE_OBJECT_INDICES = (487, 508, 620, 664, 675, 725, 734, 741, 898)


class WasteClassifier:
    """Three-class waste classifier with conservative semantic cross-checks."""

    def __init__(
        self,
        model_path=None,
        labels_path=None,
        smoothing_frames=10,
        confidence_threshold=70.0,
    ):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else self.base_dir / "model" / "keras_model.h5"
        self.labels_path = Path(labels_path) if labels_path else self.base_dir / "model" / "labels.txt"

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

        self.smoothing_frames = max(1, int(smoothing_frames))
        self.confidence_threshold = float(confidence_threshold)
        self.history_buffer = deque(maxlen=self.smoothing_frames)

        self.load_labels()
        self.load_model()

        if self.semantic_guard_enabled and TF_AVAILABLE:
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
                    parsed.append(parts[1].strip() if len(parts) > 1 and parts[0].isdigit() else line)
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
        for candidate in (self.model_path, saved_model_dir):
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
        """Load the small ImageNet recognizer used only as a second opinion."""
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
            print("[Classifier] ImageNet object cross-check is ready.")
        except Exception as error:
            self.semantic_model = None
            self.semantic_status = "unavailable"
            self.semantic_error = str(error)
            print(f"[Classifier] Object cross-check unavailable: {error}")

    def _wet_label_index(self):
        for index, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized == "wet waste" or "organic" in normalized:
                return index
        return None

    def _recyclable_label_index(self):
        for index, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized == "recyclable" or "recycle" in normalized:
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
        height, width = frame.shape[:2]
        if crop_box is None:
            base_x, base_y, base_w, base_h = 0, 0, width, height
        else:
            base_x, base_y, base_w, base_h = crop_box

        center_x = base_x + base_w / 2.0
        center_y = base_y + base_h / 2.0
        side = int(max(32, min(base_w, base_h) * float(scale)))
        side = min(side, width, height)
        x = int(round(center_x - side / 2.0))
        y = int(round(center_y - side / 2.0))
        x = max(0, min(x, width - side))
        y = max(0, min(y, height - side))
        return x, y, side, side

    def _appearance_food_hint(self, frame, crop_box):
        """Offline fallback for obvious fresh-food/fruit objects."""
        try:
            x, y, w, h = self._scaled_crop_box(frame, crop_box, 0.90)
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                return None
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            red = (
                cv2.inRange(hsv, (0, 55, 35), (18, 255, 255))
                | cv2.inRange(hsv, (165, 55, 35), (179, 255, 255))
            )
            produce_color = (
                cv2.inRange(hsv, (20, 55, 45), (45, 255, 255))
                | cv2.inRange(hsv, (45, 45, 35), (95, 255, 230))
            )
            red_ratio = float(np.mean(red > 0))
            produce_ratio = float(np.mean(produce_color > 0))

            def largest_component(mask):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                if count <= 1:
                    return 0.0, 0.0
                areas = stats[1:, cv2.CC_STAT_AREA]
                idx = int(np.argmax(areas)) + 1
                area = float(stats[idx, cv2.CC_STAT_AREA])
                box_area = float(stats[idx, cv2.CC_STAT_WIDTH] * stats[idx, cv2.CC_STAT_HEIGHT])
                total = float(mask.shape[0] * mask.shape[1])
                return area / total, area / max(1.0, box_area)

            red_component_ratio, red_fill = largest_component(red)
            produce_component_ratio, produce_fill = largest_component(produce_color)

            pale = cv2.inRange(hsv, (0, 8, 95), (45, 155, 255))
            near_red = cv2.dilate(red, np.ones((31, 31), np.uint8), iterations=1)
            pale_near_red = float(np.mean((pale > 0) & (near_red > 0)))

            red_apple_like = (
                red_ratio >= 0.015
                and red_component_ratio >= 0.008
                and red_fill >= 0.10
                and pale_near_red >= 0.040
            )
            whole_fruit_like = (
                produce_ratio >= 0.045
                and produce_component_ratio >= 0.045
                and produce_fill >= 0.20
            )
            if not (red_apple_like or whole_fruit_like):
                return None

            evidence = min(
                0.98,
                0.62
                + min(red_component_ratio / 0.12, 1.0) * 0.16
                + min(pale_near_red / 0.15, 1.0) * 0.12
                + min(max(produce_component_ratio, red_component_ratio) / 0.20, 1.0) * 0.10,
            )
            label = "apple / fresh produce" if red_apple_like else "fresh produce"
            return {
                "matched": True,
                "category": "Wet Waste",
                "label": label,
                "confidence": round(evidence * 100.0, 1),
                "produce_mass": round(max(red_ratio, produce_ratio) * 100.0, 1),
                "food_mass": round(max(red_component_ratio, produce_component_ratio) * 100.0, 1),
                "recyclable_mass": 0.0,
                "supporting_views": 1,
                "base_wet_confidence": 0.0,
                "direction": "wet",
                "source": "offline-food-appearance-guard",
            }
        except Exception as error:
            print(f"[Classifier] Offline food appearance guard skipped: {error}")
            return None

    def _semantic_hint(self, frame, crop_box, base_probs):
        """Run several center crops and collect strong food/non-food evidence."""
        appearance_hint = self._appearance_food_hint(frame, crop_box)
        if self.semantic_model is None:
            return appearance_hint

        try:
            view_scales = (0.60, 0.76, 0.90, 1.0)
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

            outputs = np.asarray(
                self.semantic_model(
                    np.stack(batches, axis=0).astype(np.float32),
                    training=False,
                ).numpy(),
                dtype=np.float32,
            )
            if outputs.ndim != 2 or outputs.shape[1] < 966:
                raise ValueError(f"Unexpected semantic output shape: {outputs.shape}")
        except Exception as error:
            self.semantic_error = str(error)
            print(f"[Classifier] Semantic inference skipped: {error}")
            return appearance_hint

        wet_index = self._wet_label_index()
        base_wet = float(base_probs[wet_index]) if wet_index is not None else 0.0

        best_produce = (None, 0.0)
        best_food = (None, 0.0)
        best_recyclable = (None, 0.0)
        max_produce_mass = 0.0
        max_food_mass = 0.0
        max_recyclable_mass = 0.0
        produce_views = 0
        recyclable_views = 0
        details = []

        for view_index, row in enumerate(outputs):
            produce_mass = float(np.sum(row[list(PRODUCE_INDICES)]))
            food_mass = float(
                np.sum(row[list(PRODUCE_INDICES)])
                + np.sum(row[list(COOKED_FOOD_INDICES)])
            )
            recyclable_mass = float(np.sum(row[list(RECYCLABLE_OBJECT_INDICES)]))

            produce_idx, produce_score = max(
                ((i, float(row[i])) for i in PRODUCE_INDICES),
                key=lambda item: item[1],
            )
            food_idx, food_score = max(
                ((i, float(row[i])) for i in COOKED_FOOD_INDICES),
                key=lambda item: item[1],
            )
            recyclable_idx, recyclable_score = max(
                ((i, float(row[i])) for i in RECYCLABLE_OBJECT_INDICES),
                key=lambda item: item[1],
            )

            if produce_score > best_produce[1]:
                best_produce = (produce_idx, produce_score)
            if food_score > best_food[1]:
                best_food = (food_idx, food_score)
            if recyclable_score > best_recyclable[1]:
                best_recyclable = (recyclable_idx, recyclable_score)

            max_produce_mass = max(max_produce_mass, produce_mass)
            max_food_mass = max(max_food_mass, food_mass)
            max_recyclable_mass = max(max_recyclable_mass, recyclable_mass)

            if produce_mass >= 0.045:
                produce_views += 1
            if recyclable_mass >= 0.12:
                recyclable_views += 1

            top_objects = sorted(
                [(i, float(row[i])) for i in set(PRODUCE_INDICES) | set(RECYCLABLE_OBJECT_INDICES)],
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            details.append({
                "scale": view_scales[view_index],
                "crop_box": boxes[view_index],
                "produce_mass": round(produce_mass * 100.0, 1),
                "food_mass": round(food_mass * 100.0, 1),
                "recyclable_mass": round(recyclable_mass * 100.0, 1),
                "top_objects": [
                    {
                        "label": IMAGENET_CLASS_MAP.get(index, str(index)),
                        "confidence": round(score * 100.0, 1),
                    }
                    for index, score in top_objects
                ],
            })

        produce_match = (
            best_produce[1] >= 0.10
            or (best_produce[1] >= 0.045 and max_produce_mass >= 0.12 and produce_views >= 2)
            or (best_produce[1] >= 0.025 and max_produce_mass >= 0.18 and base_wet >= 0.08)
        )
        cooked_match = (
            best_food[1] >= 0.18
            or (best_food[1] >= 0.09 and max_food_mass >= 0.20 and base_wet >= 0.10)
        )

        # Require strong object evidence before overriding a Wet Waste prediction.
        # This specifically handles plastic bottles, bags and common electronics.
        recyclable_match = (
            best_recyclable[1] >= 0.38
            or (best_recyclable[1] >= 0.25 and max_recyclable_mass >= 0.45 and recyclable_views >= 2)
            or (best_recyclable[1] >= 0.32 and base_wet >= 0.45)
        )

        if produce_match or cooked_match:
            candidate = best_produce if produce_match else best_food
            return {
                "matched": True,
                "category": "Wet Waste",
                "label": IMAGENET_CLASS_MAP.get(candidate[0]),
                "confidence": round(candidate[1] * 100.0, 1),
                "produce_mass": round(max_produce_mass * 100.0, 1),
                "food_mass": round(max_food_mass * 100.0, 1),
                "recyclable_mass": round(max_recyclable_mass * 100.0, 1),
                "supporting_views": produce_views,
                "base_wet_confidence": round(base_wet * 100.0, 1),
                "direction": "wet",
                "views": details,
            }

        if appearance_hint and appearance_hint.get("matched"):
            return appearance_hint

        if recyclable_match:
            return {
                "matched": True,
                "category": "Recyclable",
                "label": IMAGENET_CLASS_MAP.get(best_recyclable[0]),
                "confidence": round(best_recyclable[1] * 100.0, 1),
                "produce_mass": round(max_produce_mass * 100.0, 1),
                "food_mass": round(max_food_mass * 100.0, 1),
                "recyclable_mass": round(max_recyclable_mass * 100.0, 1),
                "supporting_views": recyclable_views,
                "base_wet_confidence": round(base_wet * 100.0, 1),
                "direction": "recyclable",
                "views": details,
            }

        return {
            "matched": False,
            "category": None,
            "label": None,
            "confidence": 0.0,
            "produce_mass": round(max_produce_mass * 100.0, 1),
            "food_mass": round(max_food_mass * 100.0, 1),
            "recyclable_mass": round(max_recyclable_mass * 100.0, 1),
            "supporting_views": 0,
            "base_wet_confidence": round(base_wet * 100.0, 1),
            "direction": None,
            "views": details,
        }

    def _apply_semantic_guard(self, raw_probs, hint):
        if not hint or not hint.get("matched"):
            return raw_probs, False

        direction = hint.get("direction")
        if direction == "wet":
            target_index = self._wet_label_index()
            minimum = 0.78
            maximum = 0.94
        elif direction == "recyclable":
            target_index = self._recyclable_label_index()
            minimum = 0.76
            maximum = 0.93
        else:
            return raw_probs, False

        if target_index is None:
            return raw_probs, False

        probs = np.asarray(raw_probs, dtype=np.float32)
        evidence = max(
            float(hint.get("confidence", 0.0)) / 100.0,
            float(hint.get("produce_mass", 0.0)) / 100.0,
            float(hint.get("recyclable_mass", 0.0)) / 100.0,
        )
        base_wet = float(hint.get("base_wet_confidence", 0.0)) / 100.0

        if direction == "wet":
            target = min(maximum, max(minimum, 0.70 + 0.48 * evidence + 0.18 * base_wet))
        else:
            target = min(maximum, max(minimum, 0.70 + 0.40 * evidence))

        remaining = 1.0 - target
        other_indices = [i for i in range(len(probs)) if i != target_index]
        other_total = float(sum(probs[i] for i in other_indices))
        if other_total > 0:
            for i in other_indices:
                probs[i] = (probs[i] / other_total) * remaining
        elif other_indices:
            each = remaining / len(other_indices)
            for i in other_indices:
                probs[i] = each

        probs[target_index] = target
        probs /= probs.sum()
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

                # IMPORTANT: run the semantic guard for ALL base classes.
                # The previous implementation skipped it when Wet Waste was
                # already the top class, which allowed a false 90%+ Wet Waste
                # prediction for a plastic bottle/electronic object.
                if self.semantic_guard_enabled:
                    if self.semantic_model is None:
                        self.load_semantic_model()
                    semantic_hint = self._semantic_hint(frame, crop_box, raw_probs)
                    raw_probs, semantic_applied = self._apply_semantic_guard(
                        raw_probs, semantic_hint
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
            direction = semantic_hint.get("direction")
            label = semantic_hint.get("label") or "recognized object"
            if direction == "wet":
                status = f"OK · OBJECT CROSS-CHECK: {label} → Wet Waste"
            elif direction == "recyclable":
                status = f"OK · OBJECT CROSS-CHECK: {label} → Recyclable"

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
