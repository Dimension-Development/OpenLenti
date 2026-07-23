"""Pre-flight TIFF validation for RIP compatibility."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import tifffile
from PIL import Image

logger = logging.getLogger(__name__)


def validate_tiff(
    path: Union[str, Path],
    expected_dpi: Optional[float] = None,
    tolerance: float = 0.01,
) -> dict:
    """Inspect a TIFF and report whether its metadata will survive a RIP without resampling."""
    path = Path(path)
    result = {
        "ok": True,
        "path": str(path),
        "dpi": None,
        "width_px": None,
        "height_px": None,
        "width_in": None,
        "height_in": None,
        "warnings": [],
        "errors": [],
    }

    if not path.exists():
        result["ok"] = False
        result["errors"].append(f"File not found: {path}")
        return result

    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            result["width_px"] = page.imagewidth
            result["height_px"] = page.imagelength

            xres = page.tags.get("XResolution")
            yres = page.tags.get("YResolution")
            unit = page.tags.get("ResolutionUnit")

            if xres is not None and yres is not None:
                def _rat(v):
                    if hasattr(v, "numerator"):
                        return float(v.numerator) / float(v.denominator) if v.denominator else float(v.numerator)
                    if isinstance(v, tuple) and len(v) == 2:
                        return float(v[0]) / float(v[1]) if v[1] else float(v[0])
                    return float(v)

                xr = _rat(xres.value)
                yr = _rat(yres.value)
                result["dpi"] = (xr, yr)

                if unit is not None and unit.value == 3:
                    xr *= 2.54
                    yr *= 2.54
                    result["dpi"] = (xr, yr)
                    result["warnings"].append("ResolutionUnit was centimetres; converted to inches")
                elif unit is not None and unit.value == 1:
                    result["warnings"].append("ResolutionUnit is 'none' - RIP may ignore DPI tags")
            else:
                result["warnings"].append("XResolution / YResolution tags missing")
    except Exception as exc:
        try:
            with Image.open(path) as img:
                result["width_px"], result["height_px"] = img.size
                dpi_info = img.info.get("dpi")
                if dpi_info:
                    result["dpi"] = tuple(float(d) for d in dpi_info)
                else:
                    result["warnings"].append("No DPI metadata found via Pillow either")
        except Exception as exc2:
            result["ok"] = False
            result["errors"].append(f"Cannot open TIFF: {exc} / {exc2}")
            return result

    if result["dpi"] and result["width_px"]:
        result["width_in"] = result["width_px"] / result["dpi"][0]
        result["height_in"] = result["height_px"] / result["dpi"][1]

    if expected_dpi is not None and result["dpi"] is not None:
        xr, yr = result["dpi"]
        if abs(xr - expected_dpi) > tolerance or abs(yr - expected_dpi) > tolerance:
            result["ok"] = False
            result["errors"].append(
                f"DPI mismatch: found ({xr:.3f}, {yr:.3f}), expected {expected_dpi:.3f}"
            )
        if abs(xr - yr) > tolerance:
            result["warnings"].append(f"Non-square resolution: X={xr:.3f} Y={yr:.3f}")

    if result["dpi"] is None:
        result["ok"] = False
        result["errors"].append("No usable DPI metadata - RIP will almost certainly resample")

    return result


def format_validation_report(report: dict) -> str:
    """Pretty-print a validation report for the CLI / GUI."""
    lines = [
        f"File          : {report['path']}",
        f"Pixel size    : {report['width_px']} x {report['height_px']}",
    ]
    if report["dpi"]:
        lines.append(f"DPI tags      : ({report['dpi'][0]:.3f}, {report['dpi'][1]:.3f})")
    else:
        lines.append("DPI tags      : <missing>")
    if report["width_in"] is not None:
        lines.append(f"Physical size : {report['width_in']:.4f} x {report['height_in']:.4f} in")
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  !  {w}")
    if report["errors"]:
        lines.append("Errors:")
        for e in report["errors"]:
            lines.append(f"  x  {e}")
    status = "PASS" if report["ok"] else "FAIL"
    lines.append(f"Result        : {status}")
    return "\n".join(lines)
