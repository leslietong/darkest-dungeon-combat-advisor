"""CLI for static screenshot ROI extraction. Does not capture or recognize."""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections.abc import Sequence
from pathlib import Path

from ddca.combat.errors import CombatModelError
from ddca.vision.calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from ddca.vision.errors import CalibrationError
from ddca.vision.extraction import MANIFEST_FILENAME, extract_regions
from ddca.vision.preview import load_bgr_image


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3B-1 extraction CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m ddca.vision.extraction_cli",
        description=(
            "Crop configured calibration ROIs from a static screenshot and write "
            "an evidence manifest. This tool does not recognize icons, read HP, "
            "or recommend actions. It never captures a live window."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Existing screenshot PNG. The file is not modified.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"YAML calibration profile (default: {DEFAULT_CALIBRATION_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for crop PNGs and manifest.json.",
    )
    parser.add_argument(
        "--region",
        action="append",
        dest="regions",
        help="Extract only this region name. Repeat to select several. Default: all profile regions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this extraction's crop files and manifest.json if they already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run static ROI extraction. Returns non-zero for invalid input."""

    _configure_windows_console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        profile = load_calibration(args.profile)
        image = load_bgr_image(args.image)
        manifest = extract_regions(
            image,
            profile,
            args.output_dir,
            selected_regions=args.regions,
            calibration_path=args.profile,
            source_image_path=args.image,
            overwrite=args.overwrite,
        )
    except (CalibrationError, CombatModelError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    manifest_path = Path(args.output_dir) / MANIFEST_FILENAME
    print(
        f"Extracted {len(manifest.regions)} region(s) from "
        f"{manifest.frame.source_image_width}x{manifest.frame.source_image_height}."
    )
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


def _configure_windows_console() -> None:
    """Print UTF-8 paths when the host console allows it."""

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


if __name__ == "__main__":
    sys.exit(main())
