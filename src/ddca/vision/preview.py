"""Draw labeled region rectangles onto a captured screenshot copy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ddca.vision.calibration import (
    VALIDATION_PROVISIONAL,
    CalibrationProfile,
)
from ddca.vision.errors import PreviewError
from ddca.vision.geometry import PixelRect

PREVIEW_GROUPS: dict[str, tuple[str, ...]] = {
    "frame": ("game_frame",),
    "ranks": (
        "hero_rank_4",
        "hero_rank_3",
        "hero_rank_2",
        "hero_rank_1",
        "enemy_rank_1",
        "enemy_rank_2",
        "enemy_rank_3",
        "enemy_rank_4",
    ),
    "health": (
        "hero_health_rank_4",
        "hero_health_rank_3",
        "hero_health_rank_2",
        "hero_health_rank_1",
        "enemy_health_rank_1",
        "enemy_health_rank_2",
        "enemy_health_rank_3",
        "enemy_health_rank_4",
    ),
    "actions": (
        "action_bar",
        "skill_slot_1",
        "skill_slot_2",
        "skill_slot_3",
        "skill_slot_4",
        "move_action_slot",
    ),
    "turn": ("round_counter", "current_hero_panel", "active_hero_marker"),
    "containers": ("combat_area", "heroes", "enemies"),
}

DEFAULT_PREVIEW_GROUPS: tuple[str, ...] = ("frame", "ranks", "health", "actions", "turn")

SHORT_LABELS: dict[str, str] = {
    "game_frame": "frame",
    "combat_area": "combat",
    "heroes": "heroes",
    "enemies": "enemies",
    "hero_rank_4": "H4",
    "hero_rank_3": "H3",
    "hero_rank_2": "H2",
    "hero_rank_1": "H1",
    "enemy_rank_1": "E1",
    "enemy_rank_2": "E2",
    "enemy_rank_3": "E3",
    "enemy_rank_4": "E4",
    "hero_health_rank_4": "HP-H4",
    "hero_health_rank_3": "HP-H3",
    "hero_health_rank_2": "HP-H2",
    "hero_health_rank_1": "HP-H1",
    "enemy_health_rank_1": "HP-E1",
    "enemy_health_rank_2": "HP-E2",
    "enemy_health_rank_3": "HP-E3?",
    "enemy_health_rank_4": "HP-E4?",
    "action_bar": "ACTIONS",
    "skill_slot_1": "S1",
    "skill_slot_2": "S2",
    "skill_slot_3": "S3",
    "skill_slot_4": "S4",
    "move_action_slot": "MOVE",
    "round_counter": "round",
    "current_hero_panel": "hero-panel",
    "active_hero_marker": "active?",
}

# BGR colors by category for contrast on Darkest Dungeon scenes.
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "frame": (200, 200, 200),
    "ranks": (80, 200, 80),
    "health": (40, 40, 255),
    "actions": (0, 220, 255),
    "turn": (0, 180, 255),
    "containers": (160, 160, 40),
    "provisional": (180, 0, 255),
}

_ENEMY_RANK_COLORS: dict[str, tuple[int, int, int]] = {
    "enemy_rank_1": (40, 90, 255),
    "enemy_rank_2": (40, 110, 230),
    "enemy_rank_3": (40, 130, 210),
    "enemy_rank_4": (40, 150, 190),
}


def names_for_preview_groups(
    profile: CalibrationProfile,
    groups: Sequence[str] | None = None,
) -> list[str]:
    """Return region names to draw for the requested preview groups."""

    selected = tuple(groups) if groups else DEFAULT_PREVIEW_GROUPS
    unknown = [name for name in selected if name not in PREVIEW_GROUPS and name != "provisional"]
    if unknown:
        known = ", ".join(sorted(PREVIEW_GROUPS) + ["provisional"])
        raise PreviewError(
            f"Unknown preview group {unknown[0]!r}. Use one of: {known}."
        )

    names: list[str] = []
    seen: set[str] = set()
    for group in selected:
        if group == "provisional":
            candidates = profile.provisional_region_names()
        else:
            candidates = PREVIEW_GROUPS[group]
        for name in candidates:
            if name in profile.regions and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def draw_calibration_preview(
    image: NDArray[np.uint8],
    profile: CalibrationProfile,
    *,
    groups: Sequence[str] | None = None,
) -> NDArray[np.uint8]:
    """Return a copy of `image` with labeled ROI rectangles.

    The original array is not modified. Container regions are omitted unless
    the `containers` group is requested. Provisional regions are dashed.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise PreviewError(
            f"Preview requires a BGR image with shape (H, W, 3); received {image.shape}. "
            "Pass a screenshot produced by python -m ddca.capture.cli capture."
        )
    canvas = np.ascontiguousarray(image.copy())
    height, width = canvas.shape[:2]
    for name in names_for_preview_groups(profile, groups):
        rect = profile.pixel_region(name, width, height)
        provisional = profile.region_validation_status(name) == VALIDATION_PROVISIONAL
        color = _color_for(name, provisional)
        _draw_labeled_rect(
            canvas,
            rect,
            SHORT_LABELS.get(name, name),
            color,
            dashed=provisional,
        )
    return canvas


def save_calibration_preview(
    image: NDArray[np.uint8],
    profile: CalibrationProfile,
    output_path: Path,
    *,
    groups: Sequence[str] | None = None,
) -> Path:
    """Draw ROIs and write a PNG. Creates the parent directory if needed."""

    canvas = draw_calibration_preview(image, profile, groups=groups)
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreviewError(
            f"Could not create preview directory {output_path.parent.resolve()}: {exc}. "
            "Choose a writable --output path."
        ) from exc

    ok = cv2.imwrite(str(output_path), canvas)
    if not ok or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise PreviewError(
            f"OpenCV failed to write preview PNG {output_path.resolve()}. "
            "Check that the path is valid, the disk is not full, and you have write permission."
        )
    return output_path


def load_bgr_image(path: Path) -> NDArray[np.uint8]:
    """Load a screenshot as a BGR array."""

    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreviewError(
            f"Could not read image {path.resolve()}. "
            "Pass a PNG created by python -m ddca.capture.cli capture "
            "and confirm the file is not corrupted."
        )
    return image


def _color_for(name: str, provisional: bool) -> tuple[int, int, int]:
    if provisional:
        return GROUP_COLORS["provisional"]
    if name in _ENEMY_RANK_COLORS:
        return _ENEMY_RANK_COLORS[name]
    for group, members in PREVIEW_GROUPS.items():
        if name in members:
            return GROUP_COLORS[group]
    return (255, 255, 255)


def _draw_labeled_rect(
    canvas: NDArray[np.uint8],
    rect: PixelRect,
    label: str,
    color: tuple[int, int, int],
    *,
    dashed: bool,
) -> None:
    if dashed:
        _draw_dashed_rect(canvas, rect, color, thickness=1)
    else:
        cv2.rectangle(
            canvas,
            (rect.left, rect.top),
            (rect.right - 1, rect.bottom - 1),
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    text_width, text_height = text_size
    text_x = min(rect.left + 2, max(0, canvas.shape[1] - text_width - 4))
    # Prefer a compact label just above short HUD boxes so it does not cover the bar.
    if rect.height <= 28:
        text_y = max(text_height + 2, rect.top - 3)
    else:
        text_y = rect.top + text_height + 2
        if text_y >= rect.bottom:
            text_y = max(text_height + 2, rect.top - 3)
    box_left = text_x - 1
    box_top = text_y - text_height - 1
    box_right = min(canvas.shape[1] - 1, text_x + text_width + 1)
    box_bottom = min(canvas.shape[0] - 1, text_y + baseline + 1)
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (box_left, max(0, box_top)),
        (box_right, box_bottom),
        (0, 0, 0),
        thickness=-1,
    )
    blended = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
    canvas[:, :] = blended
    cv2.putText(
        canvas,
        label,
        (text_x, text_y),
        font,
        scale,
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_dashed_rect(
    canvas: NDArray[np.uint8],
    rect: PixelRect,
    color: tuple[int, int, int],
    *,
    thickness: int,
    dash: int = 6,
    gap: int = 4,
) -> None:
    points = [
        ((rect.left, rect.top), (rect.right - 1, rect.top)),
        ((rect.right - 1, rect.top), (rect.right - 1, rect.bottom - 1)),
        ((rect.right - 1, rect.bottom - 1), (rect.left, rect.bottom - 1)),
        ((rect.left, rect.bottom - 1), (rect.left, rect.top)),
    ]
    for start, end in points:
        _draw_dashed_line(canvas, start, end, color, thickness, dash, gap)


def _draw_dashed_line(
    canvas: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = int(round(float(np.hypot(x2 - x1, y2 - y1))))
    if distance <= 0:
        return
    step = dash + gap
    for offset in range(0, distance, step):
        a = offset / distance
        b = min(distance, offset + dash) / distance
        p1 = (int(round(x1 + (x2 - x1) * a)), int(round(y1 + (y2 - y1) * a)))
        p2 = (int(round(x1 + (x2 - x1) * b)), int(round(y1 + (y2 - y1) * b)))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
