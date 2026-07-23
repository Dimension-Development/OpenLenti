# OpenLenti

**Open-Source Sub-Pixel Lenticular Interlacing & Pitch Calibration Engine**

Lightweight, zero-cost Python CLI and library for interlacing multi-frame artwork and generating pitch calibration test charts for ultra-high-resolution flatbed UV printers (optimised for swissQprint Kudu and similar linear-motor machines at 1080/1200 DPI).

## Features

- **Sub-pixel phase mapping** for non-integer LPI values (e.g. 75.123 LPI @ 1200 DPI)
- **Memory-efficient** TIFF generation – stays under 2 GB RSS even for multi-gigapixel canvases
- **Second-surface** horizontal flip for printing on the rear face of lenticular sheets
- **Pitch calibration charts** with labelled high-contrast bands
- **Pre-flight validation** that DPI tags survive Ergosoft / Caldera / Onyx RIPs at 100 % scale

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Generate a pitch-test target
openlenti pitch-test \
  --out ./pitch_target_1200dpi.tif \
  --dpi 1200 \
  --center-lpi 75.000 \
  --step 0.010 \
  --range 10 \
  --second-surface

# 2. Interlace artwork frames
openlenti interlace \
  --input-dir ./render_frames/ \
  --out ./final_interlaced_print.tif \
  --dpi 1200 \
  --lpi 75.120 \
  --orientation vertical \
  --second-surface

# 3. Validate the TIFF metadata
openlenti validate ./final_interlaced_print.tif --expected-dpi 1200
```

## SwissQprint Kudu Checklist

1. RIP: disable Fit-to-Media / Auto-Scale / resampling; set scale to **100.00 % Absolute**.
2. Align sheet edge parallel to the gantry with registration pins; engage vacuum.
3. Print stack:
   - Layer 1 – mirrored interlaced CMYK (second surface)
   - Layer 2 – 100 % white flood

## License

MIT
