"""Browser live-video capture helpers for Streamlit Science Fair Mode."""

import threading
import time
from dataclasses import dataclass, field

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from ai.live_decision import DecisionSnapshot, StableDecisionTracker


@dataclass
class LiveScanState:
    """Own live inference state so a committed result cannot drift on reruns."""

    tracker: StableDecisionTracker = field(
        default_factory=lambda: StableDecisionTracker(threshold=70.0, required_frames=3)
    )
    latest_result: dict | None = None
    committed_result: dict | None = None
    committed_snapshot: DecisionSnapshot | None = None

    @property
    def final_class(self):
        return self.committed_snapshot.top_class if self.committed_snapshot else None

    def update(self, result: dict) -> DecisionSnapshot:
        """Accept a frame unless a final result has already been committed."""
        if self.committed_result is not None:
            return self.committed_snapshot
        snapshot = self.tracker.update(result.get("predictions", {}))
        self.latest_result = result
        if snapshot.is_final:
            self.committed_snapshot = snapshot
            self.committed_result = dict(result)
            self.committed_result.update(
                top_class=snapshot.top_class,
                top_confidence=round(snapshot.confidence, 1),
                is_confident=True,
            )
        return snapshot

    def display_result(self):
        """Return the committed result, or the latest live result."""
        return self.committed_result or self.latest_result

    def reset(self) -> None:
        """Clear the candidate, live result, and committed decision."""
        self.tracker.reset()
        self.latest_result = None
        self.committed_result = None
        self.committed_snapshot = None


class LiveVideoProcessor(VideoProcessorBase):
    """Mirror the browser frame and retain the newest frame for throttled inference."""

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
        box_size = int(min(width, height) * 0.6)
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
        with self._lock:
            self._latest_frame = image
            self._latest_at = time.monotonic()
            self._width = width
            self._height = height
            if self._previous_at:
                elapsed = self._latest_at - self._previous_at
                if elapsed > 0:
                    self._fps = 1.0 / elapsed
            self._previous_at = self._latest_at
        return av.VideoFrame.from_ndarray(displayed, format="bgr24")

    def latest_frame(self):
        """Return a copy of the newest frame and its monotonic timestamp."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_at

    def frame_info(self):
        """Return the latest received width, height, and approximate FPS."""
        with self._lock:
            return self._width, self._height, self._fps
