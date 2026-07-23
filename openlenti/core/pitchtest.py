"""Pitch calibration test-chart generator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openlenti.core.writer import write_tiff_from_array

logger = logging.getLogger(__name__)


def _get_font(size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a readable TrueType font; fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def generate_pitch_test(
    out_path: Union[str, Path],
    dpi: float = 1200.0,
    center_lpi: float = 75.0,
    step: float = 0.010,
    range_steps: int = 10,
    second_surface: bool = False,
    band_height_in: float = 0.6,
    label_height_in: float = 0.25,
    margin_in: float = 0.3,
    width_in: float = 8.0,
) -> Path:
    """
    Generate a multi-band pitch calibration target sheet.

    Each band contains a high-contrast alternating black/white line pattern
    interlaced at a specific LPI value, with a human-readable label.
    """
    lpi_values: List[float] = [
        center_lpi + i * step
        for i in range(-range_steps, range_steps + 1)
    ]

    width_px = int(round(width_in * dpi))
    band_h_px = int(round(band_height_in * dpi))
    label_h_px = int(round(label_height_in * dpi))
    margin_px = int(round(margin_in * dpi))

    n_bands = len(lpi_values)
    total_h_px = margin_px * 2 + n_bands * (band_h_px + label_h_px)

    canvas = np.full((total_h_px, width_px, 3), 255, dtype=np.uint8)

    font = _get_font(size=max(14, int(label_h_px * 0.55)))

    y = margin_px
    for lpi in lpi_values:
        label = f"{lpi:.3f} LPI"
        label_img = Image.new("RGB", (width_px - 2 * margin_px, label_h_px), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (label_img.width - tw) // 2
        ty = (label_h_px - th) // 2
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
        label_arr = np.asarray(label_img)
        canvas[y : y + label_h_px, margin_px : margin_px + label_arr.shape[1], :] = label_arr
        y += label_h_px

        w_lens = dpi / lpi
        x = np.arange(width_px, dtype=np.float64)
        phase = np.mod(x / w_lens, 1.0)
        line = np.where(phase < 0.5, 0, 255).astype(np.uint8)
        band = np.broadcast_to(line[np.newaxis, :, np.newaxis], (band_h_px, width_px, 3)).copy()

        canvas[y : y + band_h_px, :, :] = band
        y += band_h_px

    if second_surface:
        canvas = np.ascontiguousarray(np.fliplr(canvas))

    write_tiff_from_array(out_path, canvas, dpi=(dpi, dpi))
    logger.info(
        "Pitch test written: %d bands centred on %.3f LPI \u2192 %s",
        n_bands, center_lpi, out_path,
    )
    return Path(out_path)
