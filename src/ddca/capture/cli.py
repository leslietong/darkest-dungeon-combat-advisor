"""Command-line entry point for one-shot window listing and capture."""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections.abc import Sequence
from pathlib import Path

from ddca.capture.errors import CaptureError
from ddca.capture.windows import find_window, list_visible_windows


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 capture CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m ddca.capture.cli",
        description=(
            "Read-only Darkest Dungeon window capture. "
            "This tool lists visible windows or captures one screenshot. "
            "It never injects input, reads game memory, or plays the game for you."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list",
        help="Print visible titled windows (handle, title, position, size).",
    )

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture one visible window to a timestamped PNG and JSON metadata.",
    )
    capture_parser.add_argument(
        "--title",
        required=True,
        help='Window title to match, for example "Darkest Dungeon".',
    )
    capture_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "screenshots",
        help="Directory for PNG and JSON files (default: data/screenshots).",
    )
    return parser


def cmd_list() -> int:
    """Print visible titled windows and return an exit code."""

    windows = list_visible_windows()
    if not windows:
        print("No visible windows with titles were found.")
        return 0

    print(
        f"{'HWND':<12} {'LEFT':>6} {'TOP':>6} {'WIDTH':>6} {'HEIGHT':>6} "
        f"{'STATE':<10} TITLE"
    )
    for window in windows:
        state = "minimized" if window.minimized else "visible"
        print(
            f"{window.hwnd:<12} {window.left:6d} {window.top:6d} "
            f"{window.width:6d} {window.height:6d} {state:<10} {window.title}"
        )
    return 0


def cmd_capture(title: str, output_dir: Path) -> int:
    """Find, capture, and save one window. Returns 0 on success."""

    from ddca.capture.screenshot import capture_and_save

    window = find_window(title)
    result = capture_and_save(window, output_dir)
    print(str(result.image_path.resolve()))
    print(str(result.metadata_path.resolve()))
    return 0


def _configure_windows_console() -> None:
    """Print window titles using UTF-8 when the host console allows it."""

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capture CLI. Returns a non-zero status for actionable errors."""

    _configure_windows_console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "list":
            return cmd_list()
        if args.command == "capture":
            return cmd_capture(args.title, args.output_dir)
    except CaptureError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
