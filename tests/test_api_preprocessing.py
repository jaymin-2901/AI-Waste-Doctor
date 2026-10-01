"""Regression tests for desktop/API camera framing parity."""

from api import SCAN_ZONE_RATIO, get_scan_zone


def test_api_scan_zone_matches_desktop_square_framing():
    assert get_scan_zone(640, 480) == (176, 96, 288, 288)
    assert get_scan_zone(480, 640) == (96, 176, 288, 288)
    assert get_scan_zone(640, 480, SCAN_ZONE_RATIO)[2:] == (288, 288)