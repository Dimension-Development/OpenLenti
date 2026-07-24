"""Pitch calibration chart tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from openlenti.core.interlace import compute_frame_indices
from openlenti.core.pitchtest import generate_pitch_test
from openlenti.core.validate import validate_tiff


def test_pitch_test_writes_valid_dpi(tmp_path):
    out = tmp_path / "pitch.tif"
    path = generate_pitch_test(
        out_path=out,
        dpi=300.0,  # lower DPI for fast tests
        center_lpi=75.0,
        step=0.1,
        range_steps=2,
        second_surface=False,
        width_in=1.0,
        band_height_in=0.1,
        label_height_in=0.08,
        margin_in=0.05,
    )
    assert path.exists()
    report = validate_tiff(path, expected_dpi=300.0, tolerance=0.05)
    assert report["ok"], report
    assert report["dpi"] is not None
    assert abs(report["dpi"][0] - 300.0) < 0.05


def test_pitch_band_phase_matches_interlace_n2(tmp_path):
    """Black/white bands must match 2-frame interlacing phase."""
    dpi = 100.0
    lpi = 10.0
    width_in = 0.5  # 50 px
    out = tmp_path / "pitch.tif"
    generate_pitch_test(
        out_path=out,
        dpi=dpi,
        center_lpi=lpi,
        step=1.0,
        range_steps=0,  # single band
        second_surface=False,
        width_in=width_in,
        band_height_in=0.2,
        label_height_in=0.1,
        margin_in=0.05,
    )
    arr = np.asarray(Image.open(out))
    # Find a row that is pure band (0 or 255 per channel, not label grey).
    # Scan from bottom of image for a row of only 0/255.
    band_row = None
    for y in range(arr.shape[0] - 1, -1, -1):
        row = arr[y, :, 0]
        if set(np.unique(row)).issubset({0, 255}) and 0 in row and 255 in row:
            band_row = row
            break
    assert band_row is not None, "Could not locate a high-contrast band row"

    idx = compute_frame_indices(arr.shape[1], dpi, lpi, n_frames=2)
    # frame 0 → black (0), frame 1 → white (255)
    expected = np.where(idx == 0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(band_row, expected)


def test_pitch_band_phase_offset_matches_interlace(tmp_path):
    dpi, lpi, phase = 100.0, 10.0, 0.25
    out = tmp_path / "pitch.tif"
    generate_pitch_test(
        out_path=out,
        dpi=dpi,
        center_lpi=lpi,
        step=1.0,
        range_steps=0,
        second_surface=False,
        width_in=0.5,
        band_height_in=0.2,
        label_height_in=0.1,
        margin_in=0.05,
        phase_offset=phase,
    )
    arr = np.asarray(Image.open(out))
    band_row = None
    for y in range(arr.shape[0] - 1, -1, -1):
        row = arr[y, :, 0]
        if set(np.unique(row)).issubset({0, 255}) and 0 in row and 255 in row:
            band_row = row
            break
    assert band_row is not None
    idx = compute_frame_indices(arr.shape[1], dpi, lpi, n_frames=2, phase_offset=phase)
    expected = np.where(idx == 0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(band_row, expected)


def test_second_surface_mirrors_chart(tmp_path):
    kwargs = dict(
        dpi=100.0,
        center_lpi=10.0,
        step=1.0,
        range_steps=0,
        width_in=0.4,
        band_height_in=0.15,
        label_height_in=0.08,
        margin_in=0.04,
    )
    front = tmp_path / "front.tif"
    back = tmp_path / "back.tif"
    generate_pitch_test(out_path=front, second_surface=False, **kwargs)
    generate_pitch_test(out_path=back, second_surface=True, **kwargs)
    a = np.asarray(Image.open(front))
    b = np.asarray(Image.open(back))
    np.testing.assert_array_equal(b, np.fliplr(a))
