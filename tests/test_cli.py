"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from openlenti.cli.main import app
from openlenti.core.writer import write_tiff_from_array

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "OpenLenti" in result.stdout or "OpenLenti" in result.output


def test_cli_pitch_test(tmp_path):
    out = tmp_path / "pitch.tif"
    result = runner.invoke(
        app,
        [
            "pitch-test",
            "--out",
            str(out),
            "--dpi",
            "100",
            "--center-lpi",
            "10",
            "--step",
            "0.5",
            "--range",
            "1",
            "--width",
            "2.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_interlace_and_validate(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0)]):
        Image.new("RGB", (40, 20), colour).save(frames / f"{i:02d}.png")

    out = tmp_path / "interlaced.tif"
    result = runner.invoke(
        app,
        [
            "interlace",
            "--input-dir",
            str(frames),
            "--out",
            str(out),
            "--dpi",
            "100",
            "--lpi",
            "10",
            "--orientation",
            "vertical",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    result = runner.invoke(app, ["validate", str(out), "--expected-dpi", "100"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_validate_fails_on_mismatch(tmp_path):
    path = tmp_path / "x.tif"
    write_tiff_from_array(path, np.zeros((8, 8, 3), dtype=np.uint8), dpi=(72.0, 72.0))
    result = runner.invoke(app, ["validate", str(path), "--expected-dpi", "1200"])
    assert result.exit_code == 1
    assert "FAIL" in result.output or "mismatch" in result.output.lower()


def test_cli_estimate(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        Image.new("RGB", (40, 20), (i * 50, 0, 0)).save(frames / f"{i}.png")
    result = runner.invoke(
        app,
        [
            "estimate",
            "--input-dir",
            str(frames),
            "--dpi",
            "100",
            "--lpi",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "READY" in result.output
    assert "strip" in result.output.lower() or "Lens" in result.output


def test_cli_interlace_dry_run(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        Image.new("RGB", (20, 10), (0, 0, 0)).save(frames / f"{i}.png")
    out = tmp_path / "should_not_exist.tif"
    result = runner.invoke(
        app,
        [
            "interlace",
            "--input-dir",
            str(frames),
            "--out",
            str(out),
            "--dpi",
            "100",
            "--lpi",
            "10",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "Dry run" in result.output or "READY" in result.output


def test_cli_phase_offset_flag(tmp_path):
    out = tmp_path / "pitch.tif"
    result = runner.invoke(
        app,
        [
            "pitch-test",
            "--out",
            str(out),
            "--dpi",
            "100",
            "--center-lpi",
            "10",
            "--range",
            "0",
            "--width",
            "1.0",
            "--phase-offset",
            "0.25",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
