"""Sub-pixel lenticular interlacing.

Phase mapping
-------------
One lens covers ``dpi / lpi`` device pixels (may be non-integer).  Within each
lens the N source frames are laid down as equal-width strips::

    strip_width_px = (dpi / lpi) / N
    frame_index(x) = floor( mod(x * lpi / dpi + phase_offset, 1.0) * N )

``phase_offset`` is a fraction of one lens in ``[0, 1)`` used to nudge strip
alignment after pitch calibration.

Using a continuous phase (rather than integer pixel-per-lens rounding) keeps
non-integer LPI values (e.g. 75.123 @ 1200 DPI) free of cumulative drift.

Orientation
-----------
* ``vertical``   – lenses run top→bottom; phase varies along X (3D / L-R).
* ``horizontal`` – lenses run left→right; phase varies along Y (flip).

Second-surface
--------------
Printing on the rear face of the lenticular sheet requires a mirror of the
interlaced result along the phase axis (horizontal flip for vertical lenses,
vertical flip for horizontal lenses) so the image reads correctly when viewed
from the front through the lenses.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from openlenti.core.writer import (
    finalise_memmap_tiff,
    open_memmap_writer,
    write_tiff_from_array,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}

# Prefer in-memory write below this many output bytes; use memmap above it.
_DEFAULT_MEMMAP_THRESHOLD = 1_500_000_000  # ~1.5 GB

ProgressCallback = Optional[Callable[[float, str], None]]


def natural_sort_key(path: Union[str, Path]) -> list:
    """Sort key so frame_2.png sorts before frame_10.png."""
    name = Path(path).name
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", name)
    ]


def normalize_phase_offset(phase_offset: float) -> float:
    """Wrap phase offset into ``[0, 1)``."""
    return float(np.mod(float(phase_offset), 1.0))


def compute_frame_indices(
    length: int,
    dpi: float,
    lpi: float,
    n_frames: int,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """
    Map each device pixel along the phase axis to a frame index in ``0 .. n_frames-1``.

    Parameters
    ----------
    length:
        Number of pixels along the interlacing axis (width for vertical lenses).
    dpi:
        Device resolution in dots per inch.
    lpi:
        Lens pitch in lines (lenses) per inch.  Non-integer values are supported.
    n_frames:
        Number of source frames (views).
    phase_offset:
        Fractional lens shift in ``[0, 1)`` applied before quantisation.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if dpi <= 0 or lpi <= 0:
        raise ValueError("dpi and lpi must be positive")

    offset = normalize_phase_offset(phase_offset)
    # Continuous lens phase in [0, 1), then quantise into N equal strips.
    x = np.arange(length, dtype=np.float64)
    phase = np.mod(x * (lpi / dpi) + offset, 1.0)
    # Guard the rare 1.0-epsilon case so index never equals n_frames.
    indices = np.minimum(np.floor(phase * n_frames).astype(np.intp), n_frames - 1)
    return indices


def pixels_per_lens(dpi: float, lpi: float) -> float:
    """Return the (possibly non-integer) device pixels spanned by one lens."""
    if dpi <= 0 or lpi <= 0:
        raise ValueError("dpi and lpi must be positive")
    return dpi / lpi


def estimate_job(
    width_px: int,
    height_px: int,
    n_frames: int,
    dpi: float,
    lpi: float,
    channels: int = 3,
    bytes_per_sample: int = 1,
    phase_offset: float = 0.0,
    orientation: str = "vertical",
    second_surface: bool = False,
) -> Dict[str, Any]:
    """
    Estimate physical size, strip geometry, and memory for an interlace job.

    Does not load image data.  Useful for pre-flight UI / CLI dry-runs.
    """
    if width_px < 1 or height_px < 1:
        raise ValueError("width_px and height_px must be >= 1")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if dpi <= 0 or lpi <= 0:
        raise ValueError("dpi and lpi must be positive")
    if channels < 1:
        raise ValueError("channels must be >= 1")

    orientation = orientation.lower().strip()
    ppl = pixels_per_lens(dpi, lpi)
    strip_w = ppl / n_frames
    out_bytes = int(width_px) * int(height_px) * int(channels) * int(bytes_per_sample)
    frames_bytes = out_bytes * int(n_frames)
    # Rough peak: all frames + output (conservative).
    peak_bytes = frames_bytes + out_bytes

    warnings: List[str] = []
    errors: List[str] = []
    if ppl < n_frames:
        warnings.append(
            f"Lens spans only {ppl:.3f} px for {n_frames} frames "
            f"(strip ≈ {strip_w:.3f} px) — sub-pixel strips; some views will alias"
        )
    if strip_w < 1.0:
        warnings.append(
            f"Average strip width is {strip_w:.3f} px (< 1). "
            "Consider fewer frames or higher DPI."
        )
    if out_bytes > 2 * (1024**3):
        warnings.append(
            f"Output alone is ~{out_bytes / (1024**3):.2f} GB — expect long runtimes "
            "and use sufficient free disk for memmap"
        )

    return {
        "ok": len(errors) == 0,
        "width_px": int(width_px),
        "height_px": int(height_px),
        "width_in": width_px / dpi,
        "height_in": height_px / dpi,
        "dpi": float(dpi),
        "lpi": float(lpi),
        "n_frames": int(n_frames),
        "channels": int(channels),
        "orientation": orientation,
        "second_surface": bool(second_surface),
        "phase_offset": normalize_phase_offset(phase_offset),
        "pixels_per_lens": ppl,
        "strip_width_px": strip_w,
        "output_bytes": out_bytes,
        "output_gb": out_bytes / (1024**3),
        "frames_bytes": frames_bytes,
        "frames_gb": frames_bytes / (1024**3),
        "peak_est_bytes": peak_bytes,
        "peak_est_gb": peak_bytes / (1024**3),
        "warnings": warnings,
        "errors": errors,
    }


def format_job_estimate(report: Dict[str, Any]) -> str:
    """Pretty-print an :func:`estimate_job` / :func:`preflight_frames_directory` report."""
    lines = [
        f"Frames        : {report.get('n_frames', '?')}",
        f"Pixel size    : {report.get('width_px')} x {report.get('height_px')}",
    ]
    if report.get("width_in") is not None:
        lines.append(
            f"Physical size : {report['width_in']:.4f} x {report['height_in']:.4f} in "
            f"@ {report.get('dpi', '?')} DPI"
        )
    if report.get("lpi") is not None:
        lines.append(
            f"Lens geometry : {report['lpi']:.4f} LPI → "
            f"{report.get('pixels_per_lens', 0):.3f} px/lens, "
            f"strip ≈ {report.get('strip_width_px', 0):.3f} px"
        )
    if report.get("phase_offset") is not None:
        lines.append(f"Phase offset  : {report['phase_offset']:.4f} lens")
    if report.get("orientation") is not None:
        surf = "second-surface" if report.get("second_surface") else "front-surface"
        lines.append(f"Layout        : {report['orientation']}, {surf}")
    if report.get("output_gb") is not None:
        lines.append(
            f"Output size   : ~{report['output_gb']:.3f} GB "
            f"({report.get('output_bytes', 0):,} bytes)"
        )
    if report.get("peak_est_gb") is not None:
        lines.append(
            f"Peak RAM est. : ~{report['peak_est_gb']:.3f} GB "
            f"(frames + output, rough)"
        )
    if report.get("frame_paths"):
        names = [Path(p).name for p in report["frame_paths"][:8]]
        extra = len(report["frame_paths"]) - len(names)
        listing = ", ".join(names)
        if extra > 0:
            listing += f", … (+{extra} more)"
        lines.append(f"Order         : {listing}")
    if report.get("warnings"):
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  !  {w}")
    if report.get("errors"):
        lines.append("Errors:")
        for e in report["errors"]:
            lines.append(f"  x  {e}")
    status = "READY" if report.get("ok", False) else "BLOCKED"
    lines.append(f"Pre-flight    : {status}")
    return "\n".join(lines)


def preflight_frames_directory(
    input_dir: Union[str, Path],
    dpi: float,
    lpi: float,
    orientation: str = "vertical",
    second_surface: bool = False,
    phase_offset: float = 0.0,
) -> Dict[str, Any]:
    """
    Inspect frame files (headers only) and return a job estimate + size checks.
    """
    paths = list_frame_paths(input_dir)
    report: Dict[str, Any] = {
        "ok": True,
        "path": str(input_dir),
        "frame_paths": [str(p) for p in paths],
        "n_frames": len(paths),
        "width_px": None,
        "height_px": None,
        "channels": None,
        "modes": [],
        "warnings": [],
        "errors": [],
    }

    if not paths:
        report["ok"] = False
        report["errors"].append(f"No image frames found in {input_dir}")
        return report

    sizes = []
    modes = []
    for p in paths:
        try:
            with Image.open(p) as img:
                sizes.append(img.size)  # (w, h)
                modes.append(img.mode)
        except Exception as exc:
            report["ok"] = False
            report["errors"].append(f"Cannot open {p.name}: {exc}")
            return report

    report["modes"] = modes
    unique_sizes = set(sizes)
    if len(unique_sizes) > 1:
        report["ok"] = False
        detail = ", ".join(f"{w}x{h}" for w, h in sorted(unique_sizes))
        report["errors"].append(f"Frame dimensions differ: {detail}")
        # Still fill what we can from first frame.
    w0, h0 = sizes[0]
    report["width_px"] = w0
    report["height_px"] = h0

    # Rough channel count from mode of first frame.
    mode0 = modes[0]
    channels = {"L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}.get(mode0, 3)
    if len(set(modes)) > 1:
        report["warnings"].append(
            f"Mixed colour modes among frames: {sorted(set(modes))} — "
            "they will be normalised on load"
        )
    report["channels"] = channels

    if report["ok"]:
        est = estimate_job(
            width_px=w0,
            height_px=h0,
            n_frames=len(paths),
            dpi=dpi,
            lpi=lpi,
            channels=channels,
            phase_offset=phase_offset,
            orientation=orientation,
            second_surface=second_surface,
        )
        # Merge estimate fields while keeping path/errors/warnings.
        warnings = list(report["warnings"]) + list(est.pop("warnings", []))
        errors = list(report["errors"]) + list(est.pop("errors", []))
        report.update(est)
        report["warnings"] = warnings
        report["errors"] = errors
        report["ok"] = len(errors) == 0
        report["frame_paths"] = [str(p) for p in paths]
        report["path"] = str(input_dir)

    return report


def _normalise_frame(arr: np.ndarray) -> np.ndarray:
    """Ensure HxWxC uint8 (or matching integer dtype) layout."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _validate_frame_stack(frames: Sequence[np.ndarray]) -> tuple[int, int, int, np.dtype]:
    if not frames:
        raise ValueError("At least one frame is required")
    normalised = [_normalise_frame(f) for f in frames]
    h0, w0, c0 = normalised[0].shape
    dtype = normalised[0].dtype
    for i, f in enumerate(normalised):
        if f.shape != (h0, w0, c0):
            raise ValueError(
                f"Frame {i} shape {f.shape} does not match frame 0 {(h0, w0, c0)}"
            )
        if f.dtype != dtype:
            raise ValueError(
                f"Frame {i} dtype {f.dtype} does not match frame 0 {dtype}"
            )
    return h0, w0, c0, dtype


def _emit_progress(
    callback: ProgressCallback,
    fraction: float,
    message: str = "",
) -> None:
    if callback is None:
        return
    fraction = max(0.0, min(1.0, float(fraction)))
    try:
        callback(fraction, message)
    except TypeError:
        # Allow legacy callbacks that only accept fraction.
        callback(fraction)  # type: ignore[misc,call-arg]


def interlace_frames(
    frames: Sequence[np.ndarray],
    dpi: float,
    lpi: float,
    orientation: str = "vertical",
    second_surface: bool = False,
    phase_offset: float = 0.0,
    progress_callback: ProgressCallback = None,
) -> np.ndarray:
    """
    Interlace a sequence of equally sized image arrays.

    Parameters
    ----------
    frames:
        Sequence of HxW or HxWxC arrays (same shape / dtype).
    dpi, lpi:
        Device and lens pitch.
    orientation:
        ``\"vertical\"`` or ``\"horizontal\"``.
    second_surface:
        Mirror along the phase axis after interlacing.
    phase_offset:
        Fractional lens shift in ``[0, 1)``.
    progress_callback:
        Optional ``callback(fraction, message)`` with fraction in ``[0, 1]``.
        Callables that accept only ``fraction`` are also supported.

    Returns
    -------
    np.ndarray
        Interlaced image, same shape as each input frame.
    """
    orientation = orientation.lower().strip()
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    frames_n = [_normalise_frame(f) for f in frames]
    height, width, channels = frames_n[0].shape
    _validate_frame_stack(frames_n)
    n = len(frames_n)
    offset = normalize_phase_offset(phase_offset)

    # Soft warning when strips are sub-pixel on average (still valid, just lossy).
    ppl = pixels_per_lens(dpi, lpi)
    if ppl < n:
        logger.warning(
            "Lens spans %.3f px but there are %d frames — strips are sub-pixel "
            "and some frames will be dropped or aliased",
            ppl,
            n,
        )

    out = np.empty((height, width, channels), dtype=frames_n[0].dtype)

    if orientation == "vertical":
        indices = compute_frame_indices(width, dpi, lpi, n, phase_offset=offset)
        for i, frame in enumerate(frames_n):
            mask = indices == i
            if np.any(mask):
                out[:, mask, :] = frame[:, mask, :]
            _emit_progress(
                progress_callback,
                (i + 1) / n,
                f"Interlacing frame {i + 1}/{n}",
            )
    else:
        indices = compute_frame_indices(height, dpi, lpi, n, phase_offset=offset)
        for i, frame in enumerate(frames_n):
            mask = indices == i
            if np.any(mask):
                out[mask, :, :] = frame[mask, :, :]
            _emit_progress(
                progress_callback,
                (i + 1) / n,
                f"Interlacing frame {i + 1}/{n}",
            )

    if second_surface:
        if orientation == "vertical":
            out = np.ascontiguousarray(np.fliplr(out))
        else:
            out = np.ascontiguousarray(np.flipud(out))

    _emit_progress(progress_callback, 1.0, "Interlace complete")

    logger.info(
        "Interlaced %d frames (%d x %d) @ %.3f LPI / %.1f DPI "
        "[%s%s, phase=%.4f]",
        n,
        width,
        height,
        lpi,
        dpi,
        orientation,
        ", second-surface" if second_surface else "",
        offset,
    )
    return out


def list_frame_paths(input_dir: Union[str, Path]) -> List[Path]:
    """Return image paths in a directory sorted with natural filename order."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_sort_key)


def _load_frame(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        # Preserve alpha if present; otherwise RGB / L.
        if img.mode in ("RGBA", "RGB", "L"):
            converted = img
        elif img.mode == "P":
            converted = img.convert("RGBA" if "transparency" in img.info else "RGB")
        elif img.mode in ("CMYK", "YCbCr", "LAB"):
            converted = img.convert("RGB")
        else:
            converted = img.convert("RGB")
        arr = np.asarray(converted)
    return _normalise_frame(arr)


def interlace_from_directory(
    input_dir: Union[str, Path],
    out_path: Union[str, Path],
    dpi: float,
    lpi: float,
    orientation: str = "vertical",
    second_surface: bool = False,
    phase_offset: float = 0.0,
    progress_callback: ProgressCallback = None,
    memmap_threshold_bytes: int = _DEFAULT_MEMMAP_THRESHOLD,
) -> Path:
    """
    Load ordered frames from ``input_dir``, interlace, and write a TIFF.

    Frames are sorted with natural filename order.  Large outputs are written
    via a temporary memmap; smaller jobs use an in-memory write.

    Progress spans load (0–0.35), interlace (0.35–0.90), write (0.90–1.0).
    """
    paths = list_frame_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    n = len(paths)
    logger.info("Loading %d frames from %s", n, input_dir)
    frames: List[np.ndarray] = []
    for i, p in enumerate(paths):
        frames.append(_load_frame(p))
        _emit_progress(
            progress_callback,
            0.35 * (i + 1) / n,
            f"Loading {p.name} ({i + 1}/{n})",
        )

    def _map_interlace_progress(frac: float, message: str = "") -> None:
        # Map 0..1 interlace work into 0.35..0.90 overall.
        _emit_progress(progress_callback, 0.35 + 0.55 * frac, message)

    result = interlace_frames(
        frames,
        dpi=dpi,
        lpi=lpi,
        orientation=orientation,
        second_surface=second_surface,
        phase_offset=phase_offset,
        progress_callback=_map_interlace_progress,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress_callback, 0.92, "Writing TIFF…")
    nbytes = int(result.nbytes)
    if nbytes >= memmap_threshold_bytes:
        logger.info(
            "Output is %.2f GB — writing via memmap",
            nbytes / (1024**3),
        )
        mm = open_memmap_writer(out_path, result.shape, dtype=result.dtype, dpi=(dpi, dpi))
        raw_path = out_path.with_suffix(".raw")
        try:
            mm[:] = result
            mm.flush()
            del mm
            finalise_memmap_tiff(
                raw_path,
                out_path,
                shape=result.shape,
                dtype=result.dtype,
                dpi=(dpi, dpi),
            )
        except Exception:
            try:
                if raw_path.exists():
                    raw_path.unlink()
            except OSError:
                pass
            raise
    else:
        write_tiff_from_array(out_path, result, dpi=(dpi, dpi))

    _emit_progress(progress_callback, 1.0, f"Wrote {out_path}")
    return out_path
