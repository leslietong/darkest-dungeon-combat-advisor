"""CLI for Phase 3B-2 action-slot availability. Does not identify skills."""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections.abc import Sequence
from pathlib import Path

from ddca.combat.errors import CombatModelError
from ddca.vision.action_state import (
    DEFAULT_ACTION_STATE_CONFIG_PATH,
    classify_extraction_dir,
    load_action_slot_state_config,
    write_action_slot_state_report,
)
from ddca.vision.errors import CalibrationError


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3B-2 action-slot state CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m ddca.vision.action_state_cli",
        description=(
            "Classify the four skill slots and Move from a Phase 3B-1 extraction "
            "as available, disabled, or unknown. This tool does not name skills, "
            "compute legal targets, or build a CombatState."
        ),
    )
    parser.add_argument(
        "--extraction-dir",
        type=Path,
        required=True,
        help="Phase 3B-1 extraction directory containing manifest.json and crops/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_ACTION_STATE_CONFIG_PATH,
        help=f"Versioned classifier config (default: {DEFAULT_ACTION_STATE_CONFIG_PATH.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report path. Parent directories are created if needed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the JSON report if it already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Classify five action slots. Returns non-zero for invalid input."""

    _configure_windows_console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_action_slot_state_config(args.config)
        report = classify_extraction_dir(args.extraction_dir, config)
        output_path = write_action_slot_state_report(report, args.output, overwrite=args.overwrite)
    except (CalibrationError, CombatModelError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Classified {len(report.classifications)} action slots. Config SHA-256 {report.config_sha256}.")
    print(f"Report: {output_path.resolve()}")
    print(
        f"{'slot':18} {'state':10} {'qual':6} {'chroma':7} {'margin':7} "
        f"{'conf':6} {'sha':10} reason"
    )
    for item in report.classifications:
        print(
            f"{item.slot.value:18} {item.state:10} {item.quality:6.3f} "
            f"{item.features.chroma_score:7.3f} {item.normalized_margin:7.3f} "
            f"{item.is_available.confidence.score:6.3f} {item.crop_sha256[:8]} "
            f"{item.decision_reason}"
        )
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
