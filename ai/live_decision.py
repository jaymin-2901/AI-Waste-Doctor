"""Stable live-classification decision logic shared by UI integrations."""

from dataclasses import dataclass
from typing import Mapping, Optional

@dataclass(frozen=True)
class DecisionSnapshot:
    """Current live candidate and whether it is ready to be committed."""

    top_class: str
    confidence: float
    consecutive_frames: int
    is_confident: bool
    is_final: bool


class StableDecisionTracker:
    """Require stable confidence before committing a live camera decision."""

    def __init__(self, threshold: float = 70.0, required_frames: int = 3):
        self.threshold = float(threshold)
        self.required_frames = max(1, int(required_frames))
        self.candidate_class: Optional[str] = None
        self.consecutive_frames = 0

    def reset(self) -> None:
        """Forget the current object and its accumulated confidence."""
        self.candidate_class = None
        self.consecutive_frames = 0

    def update(self, predictions: Mapping[str, float]) -> DecisionSnapshot:
        """Process one confidence frame and return the current decision state."""
        if not predictions:
            self.reset()
            return DecisionSnapshot("", 0.0, 0, False, False)

        top_class, confidence = max(predictions.items(), key=lambda item: item[1])
        confidence = float(confidence)
        if confidence < self.threshold:
            self.reset()
            return DecisionSnapshot(top_class, confidence, 0, False, False)

        if top_class == self.candidate_class:
            self.consecutive_frames += 1
        else:
            self.candidate_class = top_class
            self.consecutive_frames = 1

        return DecisionSnapshot(
            top_class,
            confidence,
            self.consecutive_frames,
            True,
            self.consecutive_frames >= self.required_frames,
        )
