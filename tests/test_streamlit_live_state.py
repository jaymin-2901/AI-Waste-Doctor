"""Tests for Streamlit live result freezing and camera telemetry."""

from live_scan import LiveScanState


def prediction(recyclable, dry=10.0, wet=5.0):
    return {
        "top_class": "Recyclable",
        "top_confidence": recyclable,
        "is_confident": recyclable >= 70,
        "predictions": {"Recyclable": recyclable, "Dry Waste": dry, "Wet Waste": wet},
    }


def test_committed_result_does_not_change_until_reset():
    state = LiveScanState()
    state.update(prediction(80.0))
    state.update(prediction(82.0))
    state.update(prediction(85.0))
    state.update(prediction(25.0, dry=70.0, wet=5.0))

    assert state.final_class == "Recyclable"
    assert state.display_result()["top_confidence"] == 85.0

    state.reset()
    assert state.display_result() is None
