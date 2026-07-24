"""TIFF writer DPI tags and validate_tiff tests."""

from __future__ import annotations

import numpy as np
import pytest

from openlenti.core.validate import format_validation_report, validate_tiff
from openlenti.core.writer import write_tiff_from_array


def test_write_and_validate_dpi_roundtrip(tmp_path):
    path = tmp_path / "rgb.tif"
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[:, :16, 0] = 255
    write_tiff_from_array(path, arr, dpi=(1200.0, 1200.0))

    report = validate_tiff(path, expected_dpi=1200.0)
    assert report["ok"], format_validation_report(report)
    assert report["width_px"] == 48
    assert report["height_px"] == 32
    assert report["dpi"] is not None
    assert abs(report["dpi"][0] - 1200.0) < 0.01
    assert abs(report["dpi"][1] - 1200.0) < 0.01
    assert report["width_in"] == pytest.approx(48 / 1200.0)
    assert report["height_in"] == pytest.approx(32 / 1200.0)


def test_validate_detects_dpi_mismatch(tmp_path):
    path = tmp_path / "m.tif"
    write_tiff_from_array(
        path,
        np.zeros((8, 8, 3), dtype=np.uint8),
        dpi=(300.0, 300.0),
    )
    report = validate_tiff(path, expected_dpi=1200.0)
    assert not report["ok"]
    assert any("DPI mismatch" in e for e in report["errors"])


def test_validate_missing_file():
    report = validate_tiff("/nonexistent/path/foo.tif")
    assert not report["ok"]
    assert report["errors"]


def test_grayscale_write(tmp_path):
    path = tmp_path / "grey.tif"
    write_tiff_from_array(path, np.arange(64, dtype=np.uint8).reshape(8, 8), dpi=(72.0, 72.0))
    report = validate_tiff(path, expected_dpi=72.0, tolerance=0.1)
    assert report["ok"], report


def test_format_validation_report_contains_status(tmp_path):
    path = tmp_path / "r.tif"
    write_tiff_from_array(path, np.zeros((4, 4, 3), dtype=np.uint8), dpi=(100.0, 100.0))
    text = format_validation_report(validate_tiff(path, expected_dpi=100.0))
    assert "PASS" in text
    assert "100" in text
