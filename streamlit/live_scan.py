import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from ai.live_decision import DecisionSnapshot, StableDecisionTracker


@dataclass
class LiveScanState:
    """Rolling stable-decision state for automatic live classification."""

    tracker: StableDecisionTracker = field(
        default_factory=lambda: StableDecisionTracker(
            threshold=70.0,
            required_frames=3,
        )
    )
    latest_result: dict | None = None
    committed_result: dict | None = None
    committed_snapshot: DecisionSnapshot | None = None

    @property
    def final_class(self):
        return (
            self.committed_snapshot.top_class
            if self.committed_snapshot
            else None
        )

    def update(self, result: dict) -> DecisionSnapshot:
        """Accept the first valid trained-model prediction as the final decision."""
        if self.committed_result is not None:
            return self.committed_snapshot

        predictions = result.get("predictions", {}) or {}
        self.latest_result = result

        if not predictions:
            return DecisionSnapshot("", 0.0, 0, False, False)

        top_class, confidence = max(
            predictions.items(),
            key=lambda item: float(item[1]),
        )
        confidence = float(confidence)

        snapshot = DecisionSnapshot(
            top_class=str(top_class),
            confidence=confidence,
            consecutive_frames=1,
            is_confident=confidence >= 70.0,
            is_final=True,
        )
        self.committed_snapshot = snapshot
        self.committed_result = dict(result)
        self.committed_result.update(
            top_class=snapshot.top_class,
            top_confidence=round(snapshot.confidence, 1),
            is_confident=snapshot.is_confident,
        )
        return snapshot

    def display_result(self):
        return self.committed_result or self.latest_result

    def reset(self) -> None:
        self.tracker.reset()
        self.latest_result = None
        self.committed_result = None
        self.committed_snapshot = None


class LiveVideoProcessor(VideoProcessorBase):
    """Receive browser camera frames and retain the newest raw frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_at = 0.0
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._previous_at = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        box_size = max(32, int(min(width, height) * 0.6))
        left = (width - box_size) // 2
        top = (height - box_size) // 2

        displayed = image.copy()
        cv2.rectangle(
            displayed,
            (left, top),
            (left + box_size, top + box_size),
            (16, 185, 129),
            3,
        )

        now = time.monotonic()
        with self._lock:
            self._latest_frame = image.copy()
            self._latest_at = now
            self._width = width
            self._height = height

            if self._previous_at:
                elapsed = now - self._previous_at
                if elapsed > 0:
                    instantaneous = 1.0 / elapsed
                    self._fps = (
                        instantaneous
                        if self._fps <= 0
                        else (0.85 * self._fps + 0.15 * instantaneous)
                    )
            self._previous_at = now

        return av.VideoFrame.from_ndarray(displayed, format="bgr24")

    def latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_at

    def frame_info(self):
        with self._lock:
            return self._width, self._height, self._fps
