"""Unit tests for sub-pixel phase mapping and interlacing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from openlenti.core.interlace import (
    compute_frame_indices,
    estimate_job,
    format_job_estimate,
    interlace_frames,
    interlace_from_directory,
    list_frame_paths,
    natural_sort_key,
    pixels_per_lens,
    preflight_frames_directory,
)


class TestComputeFrameIndices:
    def test_integer_lenses_equal_strips(self):
        # 100 DPI, 10 LPI → 10 px/lens; 2 frames → 5 px each strip.
        idx = compute_frame_indices(length=20, dpi=100.0, lpi=10.0, n_frames=2)
        assert idx.shape == (20,)
        # Lens 0: pixels 0-9 → frames 0 then 1; lens 1: 10-19 same.
        np.testing.assert_array_equal(idx[0:5], 0)
        np.testing.assert_array_equal(idx[5:10], 1)
        np.testing.assert_array_equal(idx[10:15], 0)
        np.testing.assert_array_equal(idx[15:20], 1)

    def test_non_integer_lpi_no_cumulative_drift(self):
        # 75.123 LPI @ 1200 DPI — pixels-per-lens is non-integer.
        dpi, lpi, n = 1200.0, 75.123, 4
        length = 10_000
        idx = compute_frame_indices(length, dpi, lpi, n)
        assert idx.min() == 0
        assert idx.max() == n - 1
        # Reconstruct continuous phase and check consistency.
        phase = np.mod(np.arange(length) * (lpi / dpi), 1.0)
        expected = np.minimum(np.floor(phase * n).astype(np.intp), n - 1)
        np.testing.assert_array_equal(idx, expected)

    def test_indices_cover_all_frames_when_wide_enough(self):
        idx = compute_frame_indices(length=1000, dpi=1200.0, lpi=75.0, n_frames=8)
        assert set(np.unique(idx)) == set(range(8))

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            compute_frame_indices(10, dpi=0, lpi=75, n_frames=2)
        with pytest.raises(ValueError):
            compute_frame_indices(10, dpi=1200, lpi=-1, n_frames=2)
        with pytest.raises(ValueError):
            compute_frame_indices(10, dpi=1200, lpi=75, n_frames=0)

    def test_phase_offset_shifts_indices(self):
        base = compute_frame_indices(40, dpi=100.0, lpi=10.0, n_frames=4, phase_offset=0.0)
        # Half-lens shift should move strip boundaries (not identical to base).
        shifted = compute_frame_indices(40, dpi=100.0, lpi=10.0, n_frames=4, phase_offset=0.5)
        assert not np.array_equal(base, shifted)
        # Offset of 1.0 wraps to 0.
        wrapped = compute_frame_indices(40, dpi=100.0, lpi=10.0, n_frames=4, phase_offset=1.0)
        np.testing.assert_array_equal(base, wrapped)


class TestInterlaceFrames:
    def _solid(self, h, w, value, channels=3):
        return np.full((h, w, channels), value, dtype=np.uint8)

    def test_vertical_selects_correct_frame_columns(self):
        h, w = 4, 20
        # Frame i is filled with value i.
        frames = [self._solid(h, w, i) for i in range(4)]
        dpi, lpi = 100.0, 10.0  # 10 px/lens, 4 frames → 2.5 px/strip
        out = interlace_frames(frames, dpi=dpi, lpi=lpi, orientation="vertical")
        idx = compute_frame_indices(w, dpi, lpi, 4)
        for x in range(w):
            expected = idx[x]
            assert np.all(out[:, x, :] == expected), f"col {x}"

    def test_horizontal_selects_correct_frame_rows(self):
        h, w = 20, 4
        frames = [self._solid(h, w, i) for i in range(4)]
        dpi, lpi = 100.0, 10.0
        out = interlace_frames(frames, dpi=dpi, lpi=lpi, orientation="horizontal")
        idx = compute_frame_indices(h, dpi, lpi, 4)
        for y in range(h):
            expected = idx[y]
            assert np.all(out[y, :, :] == expected), f"row {y}"

    def test_second_surface_vertical_is_fliplr(self):
        h, w = 2, 40
        frames = [self._solid(h, w, i) for i in range(2)]
        front = interlace_frames(frames, dpi=100, lpi=10, orientation="vertical", second_surface=False)
        back = interlace_frames(frames, dpi=100, lpi=10, orientation="vertical", second_surface=True)
        np.testing.assert_array_equal(back, np.fliplr(front))

    def test_second_surface_horizontal_is_flipud(self):
        h, w = 40, 2
        frames = [self._solid(h, w, i) for i in range(2)]
        front = interlace_frames(frames, dpi=100, lpi=10, orientation="horizontal", second_surface=False)
        back = interlace_frames(frames, dpi=100, lpi=10, orientation="horizontal", second_surface=True)
        np.testing.assert_array_equal(back, np.flipud(front))

    def test_grayscale_promoted_to_hxwxc(self):
        frames = [np.zeros((8, 16), dtype=np.uint8), np.full((8, 16), 255, dtype=np.uint8)]
        out = interlace_frames(frames, dpi=100, lpi=10, orientation="vertical")
        assert out.ndim == 3
        assert out.shape[2] == 1

    def test_mismatched_shapes_raise(self):
        a = np.zeros((8, 16, 3), dtype=np.uint8)
        b = np.zeros((8, 12, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            interlace_frames([a, b], dpi=100, lpi=10)

    def test_bad_orientation_raise(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2
        with pytest.raises(ValueError, match="orientation"):
            interlace_frames(frames, dpi=100, lpi=10, orientation="diagonal")

    def test_progress_callback_reaches_one(self):
        frames = [self._solid(4, 16, i) for i in range(3)]
        seen = []
        interlace_frames(
            frames,
            dpi=100,
            lpi=10,
            progress_callback=lambda f: seen.append(f),
        )
        assert seen
        assert seen[-1] == 1.0

    def test_progress_callback_two_arg_form(self):
        frames = [self._solid(4, 16, i) for i in range(2)]
        messages = []
        interlace_frames(
            frames,
            dpi=100,
            lpi=10,
            progress_callback=lambda f, m="": messages.append((f, m)),
        )
        assert messages
        assert messages[-1][0] == 1.0

    def test_phase_offset_changes_output(self):
        frames = [self._solid(4, 40, i) for i in range(4)]
        a = interlace_frames(frames, dpi=100, lpi=10, phase_offset=0.0)
        b = interlace_frames(frames, dpi=100, lpi=10, phase_offset=0.25)
        assert not np.array_equal(a, b)


class TestInterlaceFromDirectory:
    def test_roundtrip_directory(self, tmp_path):
        folder = tmp_path / "frames"
        folder.mkdir()
        # Three distinct solid frames.
        for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            img = Image.new("RGB", (32, 16), colour)
            img.save(folder / f"frame_{i:02d}.png")

        out = tmp_path / "out.tif"
        path = interlace_from_directory(
            folder,
            out,
            dpi=100.0,
            lpi=10.0,
            orientation="vertical",
            second_surface=False,
            memmap_threshold_bytes=10**18,  # force in-memory path
        )
        assert path.exists()
        arr = np.asarray(Image.open(path))
        assert arr.shape[:2] == (16, 32)

    def test_list_frame_paths_sorted(self, tmp_path):
        for name in ("b.png", "a.png", "c.jpg"):
            Image.new("RGB", (2, 2), (0, 0, 0)).save(tmp_path / name)
        (tmp_path / "notes.txt").write_text("ignore")
        paths = list_frame_paths(tmp_path)
        assert [p.name for p in paths] == ["a.png", "b.png", "c.jpg"]

    def test_natural_sort_numeric_order(self, tmp_path):
        for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
            Image.new("RGB", (2, 2), (0, 0, 0)).save(tmp_path / name)
        paths = list_frame_paths(tmp_path)
        assert [p.name for p in paths] == ["frame_1.png", "frame_2.png", "frame_10.png"]
        assert natural_sort_key("frame_2.png") < natural_sort_key("frame_10.png")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            interlace_from_directory(tmp_path, tmp_path / "o.tif", dpi=100, lpi=10)


class TestEstimateAndPreflight:
    def test_estimate_job_basic(self):
        report = estimate_job(
            width_px=1200,
            height_px=600,
            n_frames=4,
            dpi=1200.0,
            lpi=75.0,
        )
        assert report["ok"]
        assert report["width_in"] == pytest.approx(1.0)
        assert report["height_in"] == pytest.approx(0.5)
        assert report["pixels_per_lens"] == pytest.approx(16.0)
        assert report["strip_width_px"] == pytest.approx(4.0)
        assert report["output_bytes"] == 1200 * 600 * 3
        assert "READY" in format_job_estimate(report)

    def test_estimate_warns_subpixel_strips(self):
        report = estimate_job(
            width_px=100,
            height_px=50,
            n_frames=20,
            dpi=100.0,
            lpi=10.0,  # 10 px/lens / 20 frames = 0.5 px strip
        )
        assert report["ok"]
        assert report["warnings"]

    def test_preflight_matching_frames(self, tmp_path):
        for i in range(3):
            Image.new("RGB", (40, 20), (i * 40, 0, 0)).save(tmp_path / f"f{i}.png")
        report = preflight_frames_directory(tmp_path, dpi=100.0, lpi=10.0)
        assert report["ok"]
        assert report["n_frames"] == 3
        assert report["width_px"] == 40
        assert report["height_px"] == 20

    def test_preflight_mismatched_sizes(self, tmp_path):
        Image.new("RGB", (40, 20), (0, 0, 0)).save(tmp_path / "a.png")
        Image.new("RGB", (32, 20), (0, 0, 0)).save(tmp_path / "b.png")
        report = preflight_frames_directory(tmp_path, dpi=100.0, lpi=10.0)
        assert not report["ok"]
        assert any("dimensions" in e.lower() for e in report["errors"])

    def test_preflight_empty(self, tmp_path):
        report = preflight_frames_directory(tmp_path, dpi=100.0, lpi=10.0)
        assert not report["ok"]


def test_pixels_per_lens():
    assert pixels_per_lens(1200, 75) == pytest.approx(16.0)
    assert pixels_per_lens(1200, 75.123) == pytest.approx(1200 / 75.123)
