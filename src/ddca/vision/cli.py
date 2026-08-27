"""Command-line tools for YAML calibration and labeled ROI previews."""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections.abc import Sequence
from pathlib import Path

from ddca.vision.calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from ddca.vision.errors import CalibrationError
from ddca.vision.geometry import PixelRect


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 vision CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m ddca.vision.cli",
        description=(
            "Read-only calibration tools. Overlay region-of-interest rectangles "
            "on a captured screenshot or convert pixel measurements to normalized YAML. "
            "This tool never injects input or plays the game for you."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser(
        "preview",
        help="Draw labeled ROI rectangles on a captured screenshot.",
    )
    preview.add_argument(
        "--image",
        type=Path,
        required=True,
        help="PNG produced by python -m ddca.capture.cli capture.",
    )
    preview.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"YAML calibration profile (default: {DEFAULT_CALIBRATION_PATH.as_posix()}).",
    )
    preview.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "calibration_preview.png",
        help="Where to write the labeled preview PNG (default: output/calibration_preview.png).",
    )
    preview.add_argument(
        "--strict-size",
        action="store_true",
        help="Fail if the image size is not the profile's validated reference size.",
    )
    preview.add_argument(
        "--group",
        action="append",
        dest="groups",
        choices=("frame", "ranks", "health", "actions", "turn", "containers", "provisional"),
        help="ROI group to draw. Repeat to combine. Default: frame, ranks, health, actions, turn.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Load a calibration YAML and print resolved pixel rectangles.",
    )
    validate.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"YAML calibration profile (default: {DEFAULT_CALIBRATION_PATH.as_posix()}).",
    )

    from_pixels = subparsers.add_parser(
        "from-pixels",
        help="Convert a pixel rectangle into captured-window normalized YAML fields.",
    )
    from_pixels.add_argument("--left", type=int, required=True)
    from_pixels.add_argument("--top", type=int, required=True)
    from_pixels.add_argument("--width", type=int, required=True)
    from_pixels.add_argument("--height", type=int, required=True)
    from_pixels.add_argument("--image-width", type=int, required=True)
    from_pixels.add_argument("--image-height", type=int, required=True)
    return parser


def cmd_preview(
    image_path: Path,
    calibration_path: Path,
    output_path: Path,
    strict_size: bool,
    groups: Sequence[str] | None = None,
) -> int:
    """Draw and save a labeled preview. Returns 0 on success."""

    from ddca.vision.preview import load_bgr_image, save_calibration_preview

    profile = load_calibration(calibration_path)
    image = load_bgr_image(image_path)
    height, width = image.shape[:2]
    warnings = profile.size_warnings(width, height)
    if warnings and strict_size:
        print(warnings[0], file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    written = save_calibration_preview(image, profile, output_path, groups=groups)
    print(str(written.resolve()))
    return 0


def cmd_validate(calibration_path: Path) -> int:
    """Print resolved pixel rectangles at the profile reference size."""

    profile = load_calibration(calibration_path)
    print(f"profile_id: {profile.profile_id}")
    print(f"reference_size: {profile.reference_width}x{profile.reference_height}")
    print(f"display_mode: {profile.display_mode}")
    print("regions (pixel rectangles at reference size):")
    for name, rect in profile.regions.items():
        pixel = rect.to_pixel(profile.reference_width, profile.reference_height)
        status = profile.region_validation_status(name)
        print(
            f"  {name}: left={pixel.left} top={pixel.top} "
            f"width={pixel.width} height={pixel.height} status={status}"
        )
    provisional = profile.provisional_region_names()
    if provisional:
        print("provisional (not visually confirmed): " + ", ".join(provisional))
    return 0


def cmd_from_pixels(
    left: int,
    top: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> int:
    """Print normalized YAML fields for a measured pixel rectangle."""

    normalized = PixelRect(left=left, top=top, width=width, height=height).to_normalized(
        image_width,
        image_height,
    )
    print(f"x: {normalized.x:.6f}")
    print(f"y: {normalized.y:.6f}")
    print(f"width: {normalized.width:.6f}")
    print(f"height: {normalized.height:.6f}")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the vision CLI. Returns a non-zero status for actionable errors."""

    _configure_windows_console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "preview":
            return cmd_preview(
                args.image,
                args.calibration,
                args.output,
                args.strict_size,
                groups=args.groups,
            )
        if args.command == "validate":
            return cmd_validate(args.calibration)
        if args.command == "from-pixels":
            return cmd_from_pixels(
                args.left,
                args.top,
                args.width,
                args.height,
                args.image_width,
                args.image_height,
            )
    except CalibrationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
