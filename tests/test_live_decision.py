"""Tests for stable live classification decisions."""

from ai.live_decision import StableDecisionTracker


def test_confidence_below_threshold_stays_uncertain():
    tracker = StableDecisionTracker(threshold=70, required_frames=3)

    snapshot = tracker.update({"Recyclable": 69.9, "Dry Waste": 20.0})

    assert snapshot.is_confident is False
    assert snapshot.is_final is False
    assert snapshot.consecutive_frames == 0


def test_three_stable_frames_commit_final_decision():
    tracker = StableDecisionTracker(threshold=70, required_frames=3)

    snapshots = [
        tracker.update({"Recyclable": 70.0, "Dry Waste": 20.0}),
        tracker.update({"Recyclable": 82.0, "Dry Waste": 10.0}),
        tracker.update({"Recyclable": 76.0, "Dry Waste": 15.0}),
    ]

    assert [snapshot.is_final for snapshot in snapshots] == [False, False, True]
    assert snapshots[-1].top_class == "Recyclable"


def test_class_change_restarts_stability_counter():
    tracker = StableDecisionTracker(threshold=70, required_frames=2)
    tracker.update({"Recyclable": 90.0, "Dry Waste": 5.0})

    snapshot = tracker.update({"Dry Waste": 91.0, "Recyclable": 4.0})

    assert snapshot.top_class == "Dry Waste"
    assert snapshot.consecutive_frames == 1
    assert snapshot.is_final is False