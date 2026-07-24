"""OpenLenti command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from openlenti import __version__
from openlenti.core.interlace import (
    format_job_estimate,
    interlace_from_directory,
    list_frame_paths,
    preflight_frames_directory,
)
from openlenti.core.pitchtest import generate_pitch_test
from openlenti.core.validate import format_validation_report, validate_tiff

app = typer.Typer(
    name="openlenti",
    help="Sub-pixel lenticular interlacing & pitch calibration engine.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console(stderr=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"OpenLenti {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """OpenLenti — lenticular interlacing for high-DPI UV print."""
    _setup_logging(verbose)


@app.command("pitch-test")
def pitch_test_cmd(
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output TIFF path.",
        dir_okay=False,
        writable=True,
    ),
    dpi: float = typer.Option(1200.0, "--dpi", help="Device resolution (DPI)."),
    center_lpi: float = typer.Option(
        75.0,
        "--center-lpi",
        help="Centre lens pitch (LPI).",
    ),
    step: float = typer.Option(0.010, "--step", help="LPI step between bands."),
    range_steps: int = typer.Option(
        10,
        "--range",
        help="Number of steps either side of centre (total bands = 2*range+1).",
    ),
    second_surface: bool = typer.Option(
        False,
        "--second-surface/--front-surface",
        help="Mirror the chart for rear-face (second-surface) printing.",
    ),
    width_in: float = typer.Option(8.0, "--width", help="Chart width in inches."),
    phase_offset: float = typer.Option(
        0.0,
        "--phase-offset",
        help="Fractional lens phase shift in [0, 1).",
    ),
) -> None:
    """Generate a multi-band pitch calibration target sheet."""
    if dpi <= 0 or center_lpi <= 0:
        console.print("[red]DPI and centre LPI must be positive.[/red]")
        raise typer.Exit(code=1)
    if step <= 0:
        console.print("[red]Step must be positive.[/red]")
        raise typer.Exit(code=1)
    if range_steps < 0:
        console.print("[red]Range must be >= 0.[/red]")
        raise typer.Exit(code=1)

    path = generate_pitch_test(
        out_path=out,
        dpi=dpi,
        center_lpi=center_lpi,
        step=step,
        range_steps=range_steps,
        second_surface=second_surface,
        width_in=width_in,
        phase_offset=phase_offset,
    )
    console.print(f"[green]Pitch test written →[/green] {path}")


@app.command("estimate")
def estimate_cmd(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Folder of ordered frame images.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    dpi: float = typer.Option(1200.0, "--dpi", help="Device resolution (DPI)."),
    lpi: float = typer.Option(..., "--lpi", help="Lens pitch (LPI)."),
    orientation: str = typer.Option(
        "vertical",
        "--orientation",
        "-O",
        help="Lens orientation: vertical or horizontal.",
        case_sensitive=False,
    ),
    second_surface: bool = typer.Option(
        False,
        "--second-surface/--front-surface",
        help="Assume rear-face (second-surface) layout.",
    ),
    phase_offset: float = typer.Option(
        0.0,
        "--phase-offset",
        help="Fractional lens phase shift in [0, 1).",
    ),
) -> None:
    """Pre-flight: frame size check, strip geometry, and size/memory estimate."""
    orientation = orientation.lower().strip()
    if orientation not in ("vertical", "horizontal"):
        console.print("[red]orientation must be 'vertical' or 'horizontal'.[/red]")
        raise typer.Exit(code=1)
    if dpi <= 0 or lpi <= 0:
        console.print("[red]DPI and LPI must be positive.[/red]")
        raise typer.Exit(code=1)

    report = preflight_frames_directory(
        input_dir,
        dpi=dpi,
        lpi=lpi,
        orientation=orientation,
        second_surface=second_surface,
        phase_offset=phase_offset,
    )
    text = format_job_estimate(report)
    if report["ok"]:
        console.print(text, style="green")
        raise typer.Exit(code=0)
    console.print(text, style="red")
    raise typer.Exit(code=1)


@app.command("interlace")
def interlace_cmd(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Folder of ordered frame images (natural filename order).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output interlaced TIFF path.",
        dir_okay=False,
        writable=True,
    ),
    dpi: float = typer.Option(1200.0, "--dpi", help="Device resolution (DPI)."),
    lpi: float = typer.Option(..., "--lpi", help="Lens pitch (LPI), may be non-integer."),
    orientation: str = typer.Option(
        "vertical",
        "--orientation",
        "-O",
        help="Lens orientation: vertical (3D/L-R) or horizontal (flip).",
        case_sensitive=False,
    ),
    second_surface: bool = typer.Option(
        False,
        "--second-surface/--front-surface",
        help="Mirror along the phase axis for rear-face printing.",
    ),
    phase_offset: float = typer.Option(
        0.0,
        "--phase-offset",
        help="Fractional lens phase shift in [0, 1) after pitch calibration.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only run pre-flight estimate; do not interlace.",
    ),
) -> None:
    """Interlace multi-frame artwork into a print-ready TIFF."""
    orientation = orientation.lower().strip()
    if orientation not in ("vertical", "horizontal"):
        console.print("[red]orientation must be 'vertical' or 'horizontal'.[/red]")
        raise typer.Exit(code=1)
    if dpi <= 0 or lpi <= 0:
        console.print("[red]DPI and LPI must be positive.[/red]")
        raise typer.Exit(code=1)

    frames = list_frame_paths(input_dir)
    if not frames:
        console.print(f"[red]No image frames found in {input_dir}[/red]")
        raise typer.Exit(code=1)

    report = preflight_frames_directory(
        input_dir,
        dpi=dpi,
        lpi=lpi,
        orientation=orientation,
        second_surface=second_surface,
        phase_offset=phase_offset,
    )
    console.print(format_job_estimate(report))
    if not report["ok"]:
        console.print("[red]Pre-flight failed — aborting.[/red]")
        raise typer.Exit(code=1)
    if dry_run:
        console.print("[cyan]Dry run only — no output written.[/cyan]")
        raise typer.Exit(code=0)

    for w in report.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {w}")

    console.print(
        f"Interlacing [cyan]{len(frames)}[/cyan] frames @ "
        f"[cyan]{lpi}[/cyan] LPI / [cyan]{dpi}[/cyan] DPI "
        f"(phase [cyan]{phase_offset}[/cyan], [cyan]{orientation}[/cyan]"
        f"{', second-surface' if second_surface else ''}) …"
    )

    def _progress(frac: float, message: str = "") -> None:
        pct = int(round(frac * 100))
        # Carriage-return progress on a single stderr line when TTY.
        if console.is_terminal:
            console.print(f"  [{pct:3d}%] {message}", end="\r")
        elif message and pct % 25 == 0:
            console.print(f"  [{pct:3d}%] {message}")

    path = interlace_from_directory(
        input_dir=input_dir,
        out_path=out,
        dpi=dpi,
        lpi=lpi,
        orientation=orientation,
        second_surface=second_surface,
        phase_offset=phase_offset,
        progress_callback=_progress,
    )
    if console.is_terminal:
        console.print()  # clear progress line
    console.print(f"[green]Interlace complete →[/green] {path}")


@app.command("validate")
def validate_cmd(
    path: Path = typer.Argument(
        ...,
        help="TIFF file to inspect.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    expected_dpi: Optional[float] = typer.Option(
        None,
        "--expected-dpi",
        help="Fail if X/Y resolution tags differ from this value.",
    ),
    tolerance: float = typer.Option(
        0.01,
        "--tolerance",
        help="Absolute DPI tolerance for --expected-dpi.",
    ),
) -> None:
    """Pre-flight check that DPI metadata will survive a RIP at 100% scale."""
    report = validate_tiff(path, expected_dpi=expected_dpi, tolerance=tolerance)
    text = format_validation_report(report)
    if report["ok"]:
        console.print(text, style="green")
        raise typer.Exit(code=0)
    console.print(text, style="red")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
