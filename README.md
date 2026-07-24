# OpenLenti

**Open-Source Sub-Pixel Lenticular Interlacing & Pitch Calibration Engine**

Lightweight, zero-cost Python CLI **and graphical app** for interlacing multi-frame artwork and generating pitch calibration test charts for ultra-high-resolution flatbed UV printers (optimised for swissQprint Kudu and similar linear-motor machines at 1080/1200 DPI).

## Features

- **Sub-pixel phase mapping** for non-integer LPI values (e.g. 75.123 LPI @ 1200 DPI)
- **Memory-aware** TIFF generation – large canvases spill to a memmap before finalising the TIFF
- **Second-surface** mirror along the phase axis for printing on the rear face of lenticular sheets
- **Pitch calibration charts** with labelled high-contrast bands (same phase model as interlacing)
- **Pre-flight validation** that DPI tags survive Ergosoft / Caldera / Onyx RIPs at 100 % scale
- **Desktop GUI** (Flet) with live preview, progress and log

## Phase model

Within each lens the *N* source frames occupy equal-width strips. Device pixel
*x* (vertical lenses) maps to a frame index via a continuous phase so non-integer
LPI never drifts across the sheet:

```text
phase       = (x * LPI / DPI)  mod  1
frame_index = floor(phase * N)          # 0 .. N-1
```

Pitch-test black/white bands use the same definition with *N* = 2.

## Installation

```bash
# Recommended: virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Core CLI only
pip install -e .

# CLI + graphical frontend
pip install -e ".[gui]"

# Dev (pytest)
pip install -e ".[dev]"
```

## Graphical app (recommended for operators)

```bash
openlenti-gui
# or
python -m openlenti.gui
```

The GUI lets you:

- Browse a folder of ordered frames (natural sort)
- Review a live job estimate (size, strip width, RAM)
- Set DPI / LPI / phase offset / orientation / second-surface
- Interlace or pitch-test with **Save as…** and real progress
- Validate the last output **or any TIFF**
- See a live preview of the result

## CLI Quick Start

```bash
# 0. Pre-flight estimate (frame sizes, strip width, RAM)
openlenti estimate \
  --input-dir ./render_frames/ \
  --dpi 1200 \
  --lpi 75.120

# 1. Generate a pitch-test target
openlenti pitch-test \
  --out ./pitch_target_1200dpi.tif \
  --dpi 1200 \
  --center-lpi 75.000 \
  --step 0.010 \
  --range 10 \
  --second-surface

# 2. Interlace artwork frames (optional --phase-offset after calibration)
openlenti interlace \
  --input-dir ./render_frames/ \
  --out ./final_interlaced_print.tif \
  --dpi 1200 \
  --lpi 75.120 \
  --orientation vertical \
  --second-surface \
  --phase-offset 0.0

# Dry-run only:
# openlenti interlace ... --dry-run

# 3. Validate any TIFF metadata
openlenti validate ./final_interlaced_print.tif --expected-dpi 1200
```

Frames are ordered with **natural sort** (`frame_2` before `frame_10`).  
`--phase-offset` is a fraction of one lens in `[0, 1)` for fine alignment after pitch testing.


## SwissQprint Kudu Checklist

1. RIP: disable Fit-to-Media / Auto-Scale / resampling; set scale to **100.00 % Absolute**.
2. Align sheet edge parallel to the gantry with registration pins; engage vacuum.
3. Print stack:
   - Layer 1 – mirrored interlaced CMYK (second surface)
   - Layer 2 – 100 % white flood

## License

MIT
