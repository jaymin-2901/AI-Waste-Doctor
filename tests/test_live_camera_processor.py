"""Tests for browser camera frame freshness and telemetry."""

import av
import numpy as np

from live_scan import LiveVideoProcessor


def test_processor_keeps_latest_frame_and_reports_dimensions():
    processor = LiveVideoProcessor()
    source = np.zeros((360, 640, 3), dtype=np.uint8)
    output = processor.recv(av.VideoFrame.from_ndarray(source, format="bgr24"))

    frame, captured_at = processor.latest_frame()
    width, height, fps = processor.frame_info()

    assert output.width == 640 and output.height == 360
    assert frame.shape == source.shape
    assert captured_at > 0
    assert (width, height) == (640, 360)
    assert fps == 0.0