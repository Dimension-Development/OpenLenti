"""
OpenLenti GUI - Flet frontend for the lenticular interlacing engine.

Run with:
    python -m openlenti.gui
or after install:
    openlenti-gui
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import flet as ft
from PIL import Image

from openlenti import __version__
from openlenti.core.interlace import interlace_from_directory
from openlenti.core.pitchtest import generate_pitch_test
from openlenti.core.validate import validate_tiff, format_validation_report

logger = logging.getLogger("openlenti.gui")

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}


def _count_frames(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


def _make_preview_base64(path: Path, max_side: int = 720) -> Optional[str]:
    try:
        img = Image.open(path)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning("Preview failed for %s: %s", path, exc)
        return None


def main(page: ft.Page) -> None:
    page.title = f"OpenLenti {__version__}"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 16
    page.window.width = 1100
    page.window.height = 720
    page.window.min_width = 900
    page.window.min_height = 600

    frames_dir: dict = {"path": None}
    last_output: dict = {"path": None}
    busy = {"flag": False}

    status_text = ft.Text("Ready.", size=13, color=ft.Colors.BLUE_GREY_200)
    progress = ft.ProgressBar(value=0, visible=False, width=400)
    log_view = ft.ListView(expand=True, spacing=2, auto_scroll=True, height=140)
    preview_image = ft.Image(src="", fit=ft.ImageFit.CONTAIN, expand=True, visible=False)
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
        alignment=ft.alignment.center,
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
        border_radius=8,
    )

    frames_label = ft.Text("No folder selected", size=12, color=ft.Colors.BLUE_GREY_300)
    dpi_field = ft.TextField(label="DPI", value="1200", width=120, dense=True)
    lpi_field = ft.TextField(label="LPI", value="75.000", width=120, dense=True)
    center_lpi_field = ft.TextField(label="Centre LPI", value="75.000", width=120, dense=True)
    step_field = ft.TextField(label="Step", value="0.010", width=100, dense=True)
    range_field = ft.TextField(label="Range +/-", value="10", width=90, dense=True)
    orientation_dd = ft.Dropdown(
        label="Orientation",
        value="vertical",
        options=[
            ft.dropdown.Option("vertical", "Vertical (3D / left-right)"),
            ft.dropdown.Option("horizontal", "Horizontal (flip / top-bottom)"),
        ],
        width=260,
        dense=True,
    )
    second_surface_cb = ft.Checkbox(label="Second-surface (mirror)", value=True)
    width_in_field = ft.TextField(label="Width (in)", value="8.0", width=100, dense=True)

    def log(msg: str, color: str = ft.Colors.BLUE_GREY_200) -> None:
        log_view.controls.append(ft.Text(msg, size=12, color=color, selectable=True))
        page.update()

    def set_status(msg: str) -> None:
        status_text.value = msg
        page.update()

    def set_busy(is_busy: bool) -> None:
        busy["flag"] = is_busy
        progress.visible = is_busy
        progress.value = None if is_busy else 0
        for btn in (btn_interlace, btn_pitch, btn_validate, btn_browse):
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

    def on_dir_result(e: ft.FilePickerResultEvent) -> None:
        if e.path:
            p = Path(e.path)
            frames_dir["path"] = p
            frames = _count_frames(p)
            frames_label.value = f"{p.name}  -  {len(frames)} frame(s)"
            frames_label.color = ft.Colors.GREEN_300 if frames else ft.Colors.ORANGE_300
            log(f"Frames folder: {p}  ({len(frames)} images)")
            page.update()

    dir_picker = ft.FilePicker(on_result=on_dir_result)
    page.overlay.append(dir_picker)

    def browse_frames(e: ft.ControlEvent) -> None:
        dir_picker.get_directory_path(dialog_title="Select folder of ordered frames")

    def _run_job(job_fn, success_msg: str) -> None:
        if busy["flag"]:
            return
        set_busy(True)
        set_status("Working...")

        def worker():
            try:
                result_path = job_fn()
                if result_path:
                    last_output["path"] = Path(result_path)

                    def on_ok():
                        show_preview(Path(result_path))
                        log(success_msg + f" -> {result_path}", ft.Colors.GREEN_300)
                        set_status(success_msg)

                    page.run_thread(on_ok)
                else:
                    page.run_thread(lambda: set_status("Done."))
            except Exception as exc:
                tb = traceback.format_exc()

                def on_err():
                    log(f"Error: {exc}", ft.Colors.RED_300)
                    log(tb, ft.Colors.RED_200)
                    set_status(f"Failed: {exc}")

                page.run_thread(on_err)
            finally:
                page.run_thread(lambda: set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def do_interlace(e: ft.ControlEvent) -> None:
        folder = frames_dir["path"]
        if not folder:
            log("Select a frames folder first.", ft.Colors.ORANGE_300)
            return
        frames = _count_frames(Path(folder))
        if not frames:
            log("No image frames found in that folder.", ft.Colors.ORANGE_300)
            return
        try:
            dpi = float(dpi_field.value)
            lpi = float(lpi_field.value)
        except ValueError:
            log("DPI / LPI must be numbers.", ft.Colors.RED_300)
            return
        out = Path(folder) / "interlaced_output.tif"

        def job():
            log(f"Interlacing {len(frames)} frames @ {lpi} LPI / {dpi} DPI ...")
            return interlace_from_directory(
                input_dir=folder,
                out_path=out,
                dpi=dpi,
                lpi=lpi,
                orientation=orientation_dd.value or "vertical",
                second_surface=bool(second_surface_cb.value),
            )

        _run_job(job, "Interlace complete")

    def do_pitch_test(e: ft.ControlEvent) -> None:
        try:
            dpi = float(dpi_field.value)
            center = float(center_lpi_field.value)
            step = float(step_field.value)
            rng = int(range_field.value)
            width_in = float(width_in_field.value)
        except ValueError:
            log("Pitch-test fields must be valid numbers.", ft.Colors.RED_300)
            return
        out = Path(tempfile.gettempdir()) / "openlenti_pitch_test.tif"

        def job():
            log(f"Generating pitch test centred on {center} LPI ...")
            return generate_pitch_test(
                out_path=out,
                dpi=dpi,
                center_lpi=center,
                step=step,
                range_steps=rng,
                second_surface=bool(second_surface_cb.value),
                width_in=width_in,
            )

        _run_job(job, "Pitch test written")

    def do_validate(e: ft.ControlEvent) -> None:
        path = last_output["path"]
        if not path or not Path(path).exists():
            log("No output TIFF yet - interlace or generate a pitch test first.", ft.Colors.ORANGE_300)
            return
        try:
            dpi = float(dpi_field.value)
        except ValueError:
            dpi = None

        def job():
            report = validate_tiff(path, expected_dpi=dpi)
            text = format_validation_report(report)
            color = ft.Colors.GREEN_300 if report["ok"] else ft.Colors.RED_300
            log(text, color)
            return path

        _run_job(job, "Validation finished")

    btn_browse = ft.ElevatedButton("Browse frames folder...", icon=ft.Icons.FOLDER_OPEN, on_click=browse_frames)
    btn_interlace = ft.ElevatedButton("Interlace", icon=ft.Icons.VIEW_COLUMN, bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE, on_click=do_interlace)
    btn_pitch = ft.ElevatedButton("Pitch Test", icon=ft.Icons.STRAIGHTEN, on_click=do_pitch_test)
    btn_validate = ft.OutlinedButton("Validate last TIFF", icon=ft.Icons.VERIFIED, on_click=do_validate)

    controls_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("OpenLenti", size=22, weight=ft.FontWeight.BOLD),
                ft.Text(f"v{__version__}  -  Sub-pixel lenticular engine", size=11, color=ft.Colors.BLUE_GREY_400),
                ft.Divider(),
                ft.Text("Frames", weight=ft.FontWeight.W_600),
                btn_browse,
                frames_label,
                ft.Divider(),
                ft.Text("Interlace settings", weight=ft.FontWeight.W_600),
                ft.Row([dpi_field, lpi_field], spacing=8),
                orientation_dd,
                second_surface_cb,
                btn_interlace,
                ft.Divider(),
                ft.Text("Pitch calibration", weight=ft.FontWeight.W_600),
                ft.Row([center_lpi_field, step_field, range_field], spacing=8),
                width_in_field,
                btn_pitch,
                ft.Divider(),
                btn_validate,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        width=300,
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
                ft.Row([status_text, progress], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
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
        padding=ft.padding.only(top=8),
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
    log("OpenLenti GUI ready. Select a frames folder or generate a pitch test.")


def run_gui() -> None:
    ft.app(target=main)


if __name__ == "__main__":
    run_gui()
