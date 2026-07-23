"""Memory-efficient TIFF writer using tifffile."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def write_tiff_from_array(
    path: Union[str, Path],
    array: np.ndarray,
    dpi: Tuple[float, float] = (1200.0, 1200.0),
    compression: Optional[str] = None,
) -> None:
    """
    Write a complete in-memory array as a TIFF with correct DPI tags.

    For giant canvases that would exceed the 2 GB RAM ceiling prefer the
    memmap helpers below.
    """
    path = Path(path)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    height, width, channels = array.shape

    photometric = {1: "minisblack", 3: "rgb", 4: "separated"}.get(channels, "rgb")

    tifffile.imwrite(
        str(path),
        array,
        photometric=photometric,
        resolution=dpi,
        resolutionunit="INCH",
        compression=compression,
        metadata=None,
        bigtiff=True if (height * width * channels > 2**31) else False,
    )
    logger.info("Wrote %s (%d x %d @ %.1f DPI)", path, width, height, dpi[0])


def open_memmap_writer(
    path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype: np.dtype = np.uint8,
    dpi: Tuple[float, float] = (1200.0, 1200.0),
) -> np.memmap:
    """
    Create a memory-mapped raw array on disk.

    The returned memmap can be filled row-by-row; the caller is responsible
    for flushing it and for calling `finalise_memmap_tiff` afterwards.
    """
    path = Path(path)
    raw_path = path.with_suffix(".raw")
    memmap = np.memmap(str(raw_path), dtype=dtype, mode="w+", shape=shape)
    return memmap


def finalise_memmap_tiff(
    raw_path: Union[str, Path],
    tiff_path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    dpi: Tuple[float, float] = (1200.0, 1200.0),
) -> None:
    """Convert a raw memmap file into a proper TIFF with DPI tags."""
    raw_path = Path(raw_path)
    tiff_path = Path(tiff_path)
    height, width, channels = shape
    photometric = {1: "minisblack", 3: "rgb", 4: "separated"}.get(channels, "rgb")

    data = np.memmap(str(raw_path), dtype=dtype, mode="r", shape=shape)
    tifffile.imwrite(
        str(tiff_path),
        data,
        photometric=photometric,
        resolution=dpi,
        resolutionunit="INCH",
        bigtiff=True,
    )
    try:
        raw_path.unlink()
    except OSError:
        pass
    logger.info("Finalised TIFF %s from memmap", tiff_path)
