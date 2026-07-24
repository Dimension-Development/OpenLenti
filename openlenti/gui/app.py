"""
OpenLenti GUI - Flet frontend for the lenticular interlacing engine.

Compatible with Flet 0.24+ (tested against 0.86).

Run with:
    python -m openlenti.gui
or after install:
    openlenti-gui
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import flet as ft
from PIL import Image

from openlenti import __version__
from openlenti.core.interlace import (
    format_job_estimate,
    interlace_from_directory,
    list_frame_paths,
    preflight_frames_directory,
)
from openlenti.core.pitchtest import generate_pitch_test
from openlenti.core.validate import format_validation_report, validate_tiff

logger = logging.getLogger("openlenti.gui")

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}
TIFF_EXTENSIONS = ["tif", "tiff"]


def _count_frames(folder: Path) -> List[Path]:
    try:
        return list_frame_paths(folder)
    except (NotADirectoryError, OSError):
        return []


def _make_preview_base64(path: Path, max_side: int = 720) -> Optional[str]:
    """Build a small preview without loading full multi-gigapixel data when possible."""
    try:
        with Image.open(path) as img:
            # Prefer draft decode for JPEG; for TIFF thumbnail still may load strips.
            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass
            img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning("Preview failed for %s: %s", path, exc)
        return None


def _box_fit_contain():
    if hasattr(ft, "BoxFit"):
        return ft.BoxFit.CONTAIN
    if hasattr(ft, "ImageFit"):
        return ft.ImageFit.CONTAIN
    return None


def _alignment_center():
    if hasattr(ft, "Alignment") and hasattr(ft.Alignment, "CENTER"):
        return ft.Alignment.CENTER
    alignment = getattr(ft, "alignment", None)
    if alignment is not None and hasattr(alignment, "center"):
        return alignment.center
    return None


def _padding_only(**kwargs):
    if hasattr(ft, "Padding") and hasattr(ft.Padding, "only"):
        return ft.Padding.only(**kwargs)
    padding = getattr(ft, "padding", None)
    if padding is not None and hasattr(padding, "only"):
        return padding.only(**kwargs)
    return None


def _file_picker_type_custom():
    if hasattr(ft, "FilePickerFileType"):
        return ft.FilePickerFileType.CUSTOM
    return None


def main(page: ft.Page) -> None:
    page.title = f"OpenLenti {__version__}"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 16
    page.window.width = 1180
    page.window.height = 780
    page.window.min_width = 960
    page.window.min_height = 640

    frames_dir: dict = {"path": None}
    last_output: dict = {"path": None}
    busy = {"flag": False}

    status_text = ft.Text("Ready.", size=13, color=ft.Colors.BLUE_GREY_200)
    progress_label = ft.Text("", size=11, color=ft.Colors.BLUE_GREY_400)
    progress = ft.ProgressBar(value=0, visible=False, width=420)
    log_view = ft.ListView(expand=True, spacing=2, auto_scroll=True, height=140)
    preview_image = ft.Image(
        src="",
        fit=_box_fit_contain(),
        expand=True,
        visible=False,
    )
    preview_placeholder = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.IMAGE_OUTLINED, size=64, color=ft.Colors.BLUE_GREY_700),
                ft.Text("Preview will appear here", color=ft.Colors.BLUE_GREY_500),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        expand=True,
        alignment=_alignment_center(),
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
        border_radius=8,
    )

    frames_label = ft.Text("No folder selected", size=12, color=ft.Colors.BLUE_GREY_300)
    estimate_text = ft.Text(
        "Select a frames folder to see job estimate.",
        size=11,
        color=ft.Colors.BLUE_GREY_400,
        selectable=True,
    )

    dpi_field = ft.TextField(label="DPI", value="1200", width=110, dense=True)
    lpi_field = ft.TextField(label="LPI", value="75.000", width=110, dense=True)
    phase_field = ft.TextField(
        label="Phase (0–1)",
        value="0.000",
        width=110,
        dense=True,
        tooltip="Fractional lens shift after pitch calibration",
    )
    center_lpi_field = ft.TextField(label="Centre LPI", value="75.000", width=110, dense=True)
    step_field = ft.TextField(label="Step", value="0.010", width=90, dense=True)
    range_field = ft.TextField(label="Range +/-", value="10", width=90, dense=True)
    orientation_dd = ft.Dropdown(
        label="Orientation",
        value="vertical",
        options=[
            ft.dropdown.Option(key="vertical", text="Vertical (3D / left-right)"),
            ft.dropdown.Option(key="horizontal", text="Horizontal (flip / top-bottom)"),
        ],
        width=260,
        dense=True,
    )
    second_surface_cb = ft.Checkbox(label="Second-surface (mirror)", value=True)
    width_in_field = ft.TextField(label="Width (in)", value="8.0", width=100, dense=True)

    # Buttons declared early so closures can reference them; assigned below.
    btn_browse = ft.Button(content="Browse frames…", icon=ft.Icons.FOLDER_OPEN)
    btn_estimate = ft.OutlinedButton(content="Refresh estimate", icon=ft.Icons.CALCULATE)
    btn_interlace = ft.Button(
        content="Interlace…",
        icon=ft.Icons.VIEW_COLUMN,
        bgcolor=ft.Colors.BLUE_700,
        color=ft.Colors.WHITE,
    )
    btn_pitch = ft.Button(content="Pitch Test…", icon=ft.Icons.STRAIGHTEN)
    btn_validate_last = ft.OutlinedButton(
        content="Validate last",
        icon=ft.Icons.VERIFIED,
    )
    btn_validate_any = ft.OutlinedButton(
        content="Validate TIFF…",
        icon=ft.Icons.FIND_IN_PAGE,
    )

    action_buttons = (
        btn_browse,
        btn_estimate,
        btn_interlace,
        btn_pitch,
        btn_validate_last,
        btn_validate_any,
    )

    def _ui(fn) -> None:
        try:
            page.run_thread(fn)
        except Exception:
            fn()

    def log(msg: str, color: str = ft.Colors.BLUE_GREY_200) -> None:
        log_view.controls.append(ft.Text(msg, size=12, color=color, selectable=True))
        page.update()

    def set_status(msg: str) -> None:
        status_text.value = msg
        page.update()

    def set_progress(frac: Optional[float], message: str = "") -> None:
        """Update determinate progress. ``None`` => indeterminate."""
        progress.visible = True
        progress.value = None if frac is None else max(0.0, min(1.0, float(frac)))
        progress_label.value = message
        page.update()

    def set_busy(is_busy: bool) -> None:
        busy["flag"] = is_busy
        if not is_busy:
            progress.visible = False
            progress.value = 0
            progress_label.value = ""
        else:
            progress.visible = True
            progress.value = None
        for btn in action_buttons:
            btn.disabled = is_busy
        page.update()

    def show_preview(path: Path) -> None:
        data_uri = _make_preview_base64(path)
        if data_uri:
            preview_image.src = data_uri
            preview_image.visible = True
            preview_placeholder.visible = False
        else:
            preview_image.visible = False
            preview_placeholder.visible = True
        page.update()

    def _parse_common() -> Optional[dict]:
        try:
            dpi = float(dpi_field.value)
            lpi = float(lpi_field.value)
            phase = float(phase_field.value)
        except (TypeError, ValueError):
            log("DPI / LPI / Phase must be numbers.", ft.Colors.RED_300)
            return None
        if dpi <= 0 or lpi <= 0:
            log("DPI and LPI must be positive.", ft.Colors.RED_300)
            return None
        return {
            "dpi": dpi,
            "lpi": lpi,
            "phase_offset": phase,
            "orientation": orientation_dd.value or "vertical",
            "second_surface": bool(second_surface_cb.value),
        }

    def refresh_estimate(e=None) -> None:
        folder = frames_dir["path"]
        if not folder:
            estimate_text.value = "Select a frames folder to see job estimate."
            estimate_text.color = ft.Colors.BLUE_GREY_400
            page.update()
            return
        params = _parse_common()
        if not params:
            return
        report = preflight_frames_directory(
            folder,
            dpi=params["dpi"],
            lpi=params["lpi"],
            orientation=params["orientation"],
            second_surface=params["second_surface"],
            phase_offset=params["phase_offset"],
        )
        estimate_text.value = format_job_estimate(report)
        estimate_text.color = (
            ft.Colors.GREEN_300 if report["ok"] else ft.Colors.ORANGE_300
        )
        page.update()
        if report.get("warnings"):
            for w in report["warnings"]:
                log(f"Estimate warning: {w}", ft.Colors.ORANGE_300)
        if report.get("errors"):
            for err in report["errors"]:
                log(f"Estimate error: {err}", ft.Colors.RED_300)

    def _apply_frames_folder(path_str: str) -> None:
        p = Path(path_str)
        frames_dir["path"] = p
        frames = _count_frames(p)
        frames_label.value = f"{p.name}  —  {len(frames)} frame(s)"
        frames_label.color = ft.Colors.GREEN_300 if frames else ft.Colors.ORANGE_300
        log(f"Frames folder: {p}  ({len(frames)} images, natural sort)")
        page.update()
        refresh_estimate()

    async def browse_frames(e=None) -> None:
        try:
            path = await ft.FilePicker().get_directory_path(
                dialog_title="Select folder of ordered frames"
            )
        except Exception as exc:
            log(f"Folder picker failed: {exc}", ft.Colors.RED_300)
            return
        if path:
            _apply_frames_folder(path)

    def _run_job(job_fn, success_msg: str, start_msg: Optional[str] = None) -> None:
        if busy["flag"]:
            return
        set_busy(True)
        set_status("Working…")
        if start_msg:
            log(start_msg)

        def on_progress(frac: float, message: str = "") -> None:
            def _():
                set_progress(frac, message)
                if message:
                    set_status(message)

            _ui(_)

        def worker():
            try:
                result_path = job_fn(on_progress)
                if result_path:
                    last_output["path"] = Path(result_path)

                    def on_ok():
                        show_preview(Path(result_path))
                        log(success_msg + f" → {result_path}", ft.Colors.GREEN_300)
                        set_status(success_msg)

                    _ui(on_ok)
                else:
                    _ui(lambda: set_status("Done."))
            except Exception as exc:
                tb = traceback.format_exc()

                def on_err():
                    log(f"Error: {exc}", ft.Colors.RED_300)
                    log(tb, ft.Colors.RED_200)
                    set_status(f"Failed: {exc}")

                _ui(on_err)
            finally:
                _ui(lambda: set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    async def do_interlace(e=None) -> None:
        folder = frames_dir["path"]
        if not folder:
            log("Select a frames folder first.", ft.Colors.ORANGE_300)
            return
        frames = _count_frames(Path(folder))
        if not frames:
            log("No image frames found in that folder.", ft.Colors.ORANGE_300)
            return
        params = _parse_common()
        if not params:
            return

        # Pre-flight before save dialog.
        report = preflight_frames_directory(
            folder,
            dpi=params["dpi"],
            lpi=params["lpi"],
            orientation=params["orientation"],
            second_surface=params["second_surface"],
            phase_offset=params["phase_offset"],
        )
        estimate_text.value = format_job_estimate(report)
        estimate_text.color = (
            ft.Colors.GREEN_300 if report["ok"] else ft.Colors.ORANGE_300
        )
        page.update()
        if not report["ok"]:
            log("Pre-flight failed — fix frame issues before interlacing.", ft.Colors.RED_300)
            for err in report.get("errors") or []:
                log(f"  x  {err}", ft.Colors.RED_300)
            return
        for w in report.get("warnings") or []:
            log(f"Warning: {w}", ft.Colors.ORANGE_300)

        try:
            out = await ft.FilePicker().save_file(
                dialog_title="Save interlaced TIFF",
                file_name="interlaced_output.tif",
                file_type=_file_picker_type_custom(),
                allowed_extensions=TIFF_EXTENSIONS,
                initial_directory=str(folder),
            )
        except Exception as exc:
            log(f"Save dialog failed: {exc}", ft.Colors.RED_300)
            return
        if not out:
            log("Interlace cancelled — no output path.", ft.Colors.BLUE_GREY_400)
            return
        out_path = Path(out)
        if out_path.suffix.lower() not in {".tif", ".tiff"}:
            out_path = out_path.with_suffix(".tif")

        def job(on_progress):
            return interlace_from_directory(
                input_dir=folder,
                out_path=out_path,
                dpi=params["dpi"],
                lpi=params["lpi"],
                orientation=params["orientation"],
                second_surface=params["second_surface"],
                phase_offset=params["phase_offset"],
                progress_callback=on_progress,
            )

        _run_job(
            job,
            "Interlace complete",
            start_msg=(
                f"Interlacing {len(frames)} frames @ {params['lpi']} LPI / "
                f"{params['dpi']} DPI (phase {params['phase_offset']}) …"
            ),
        )

    async def do_pitch_test(e=None) -> None:
        try:
            dpi = float(dpi_field.value)
            center = float(center_lpi_field.value)
            step = float(step_field.value)
            rng = int(range_field.value)
            width_in = float(width_in_field.value)
            phase = float(phase_field.value)
        except (TypeError, ValueError):
            log("Pitch-test fields must be valid numbers.", ft.Colors.RED_300)
            return

        try:
            out = await ft.FilePicker().save_file(
                dialog_title="Save pitch test TIFF",
                file_name="openlenti_pitch_test.tif",
                file_type=_file_picker_type_custom(),
                allowed_extensions=TIFF_EXTENSIONS,
            )
        except Exception as exc:
            log(f"Save dialog failed: {exc}", ft.Colors.RED_300)
            return
        if not out:
            log("Pitch test cancelled — no output path.", ft.Colors.BLUE_GREY_400)
            return
        out_path = Path(out)
        if out_path.suffix.lower() not in {".tif", ".tiff"}:
            out_path = out_path.with_suffix(".tif")

        second_surface = bool(second_surface_cb.value)

        def job(on_progress):
            on_progress(0.1, "Generating pitch bands…")
            path = generate_pitch_test(
                out_path=out_path,
                dpi=dpi,
                center_lpi=center,
                step=step,
                range_steps=rng,
                second_surface=second_surface,
                width_in=width_in,
                phase_offset=phase,
            )
            on_progress(1.0, "Pitch test written")
            return path

        _run_job(
            job,
            "Pitch test written",
            start_msg=f"Generating pitch test centred on {center} LPI …",
        )

    def _validate_path(path: Path) -> None:
        try:
            dpi = float(dpi_field.value)
        except (TypeError, ValueError):
            dpi = None

        def job(on_progress):
            on_progress(0.2, f"Validating {path.name}…")
            report = validate_tiff(path, expected_dpi=dpi)
            text = format_validation_report(report)
            color = ft.Colors.GREEN_300 if report["ok"] else ft.Colors.RED_300

            def on_report():
                log(text, color)

            _ui(on_report)
            on_progress(1.0, "Validation finished")
            return path

        _run_job(job, "Validation finished", start_msg=f"Validating {path} …")

    def do_validate_last(e=None) -> None:
        path = last_output["path"]
        if not path or not Path(path).exists():
            log(
                "No output TIFF yet — interlace, pitch test, or Validate TIFF…",
                ft.Colors.ORANGE_300,
            )
            return
        _validate_path(Path(path))

    async def do_validate_any(e=None) -> None:
        try:
            files = await ft.FilePicker().pick_files(
                dialog_title="Select TIFF to validate",
                allow_multiple=False,
                file_type=_file_picker_type_custom(),
                allowed_extensions=TIFF_EXTENSIONS,
            )
        except Exception as exc:
            log(f"File picker failed: {exc}", ft.Colors.RED_300)
            return
        if not files:
            log("Validate cancelled.", ft.Colors.BLUE_GREY_400)
            return
        chosen = files[0]
        path_str = getattr(chosen, "path", None)
        if not path_str:
            log(
                "Selected file has no filesystem path (web mode?).",
                ft.Colors.ORANGE_300,
            )
            return
        path = Path(path_str)
        last_output["path"] = path
        _validate_path(path)

    # Wire handlers
    btn_browse.on_click = browse_frames
    btn_estimate.on_click = refresh_estimate
    btn_interlace.on_click = do_interlace
    btn_pitch.on_click = do_pitch_test
    btn_validate_last.on_click = do_validate_last
    btn_validate_any.on_click = do_validate_any

    # Refresh estimate when key fields change
    for field in (dpi_field, lpi_field, phase_field):
        field.on_blur = refresh_estimate
    orientation_dd.on_change = refresh_estimate
    second_surface_cb.on_change = refresh_estimate

    controls_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("OpenLenti", size=22, weight=ft.FontWeight.BOLD),
                ft.Text(
                    f"v{__version__}  —  Sub-pixel lenticular engine",
                    size=11,
                    color=ft.Colors.BLUE_GREY_400,
                ),
                ft.Divider(),
                ft.Text("Frames", weight=ft.FontWeight.W_600),
                btn_browse,
                frames_label,
                btn_estimate,
                ft.Container(
                    content=estimate_text,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                    border_radius=6,
                    padding=8,
                ),
                ft.Divider(),
                ft.Text("Interlace settings", weight=ft.FontWeight.W_600),
                ft.Row([dpi_field, lpi_field], spacing=8),
                phase_field,
                orientation_dd,
                second_surface_cb,
                btn_interlace,
                ft.Divider(),
                ft.Text("Pitch calibration", weight=ft.FontWeight.W_600),
                ft.Row([center_lpi_field, step_field, range_field], spacing=8),
                width_in_field,
                btn_pitch,
                ft.Divider(),
                ft.Text("Validate", weight=ft.FontWeight.W_600),
                ft.Row([btn_validate_last, btn_validate_any], spacing=8, wrap=True),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        width=320,
        padding=12,
        bgcolor=ft.Colors.with_opacity(0.25, ft.Colors.BLUE_GREY_900),
        border_radius=10,
    )

    preview_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Preview", weight=ft.FontWeight.W_600),
                ft.Stack([preview_placeholder, preview_image], expand=True),
            ],
            expand=True,
        ),
        expand=True,
        padding=12,
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.BLACK),
    )

    bottom_bar = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Column([status_text, progress_label], spacing=2, expand=True),
                        progress,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Container(
                    content=log_view,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                    border_radius=6,
                    padding=8,
                    height=150,
                ),
            ],
            spacing=6,
        ),
        padding=_padding_only(top=8),
    )

    page.add(
        ft.Row(
            [controls_panel, preview_panel],
            expand=True,
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
        bottom_bar,
    )
    log("OpenLenti GUI ready. Select frames, review estimate, then Interlace… or Pitch Test…")


def run_gui() -> None:
    if hasattr(ft, "run"):
        ft.run(main)
    else:
        ft.app(target=main)


if __name__ == "__main__":
    run_gui()
