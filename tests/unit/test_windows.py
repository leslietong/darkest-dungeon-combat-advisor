"""Unit tests for window matching and capture eligibility."""

from __future__ import annotations

import pytest

from ddca.capture.errors import (
    AmbiguousWindowError,
    InvalidWindowGeometryError,
    MinimizedWindowError,
    WindowNotFoundError,
)
from ddca.capture.windows import (
    WindowInfo,
    assert_window_capturable,
    find_window,
    list_visible_windows,
)


def make_window(**overrides: object) -> WindowInfo:
    values: dict[str, object] = {
        "hwnd": 1,
        "title": "Darkest Dungeon",
        "left": 0,
        "top": 0,
        "width": 1920,
        "height": 1080,
        "minimized": False,
    }
    values.update(overrides)
    return WindowInfo(
        hwnd=int(values["hwnd"]),
        title=str(values["title"]),
        left=int(values["left"]),
        top=int(values["top"]),
        width=int(values["width"]),
        height=int(values["height"]),
        minimized=bool(values["minimized"]),
    )


def test_case_insensitive_exact_match() -> None:
    windows = [make_window(title="Darkest Dungeon")]

    found = find_window("darkest dungeon", windows)

    assert found.hwnd == 1
    assert found.title == "Darkest Dungeon"


def test_unique_substring_match() -> None:
    windows = [
        make_window(hwnd=10, title="Notepad"),
        make_window(hwnd=11, title="Darkest Dungeon"),
    ]

    found = find_window("darkest", windows)

    assert found.hwnd == 11


def test_exact_match_preferred_over_substring_matches() -> None:
    windows = [
        make_window(hwnd=21, title="Darkest Dungeon Tools"),
        make_window(hwnd=22, title="Darkest Dungeon"),
        make_window(hwnd=23, title="Darkest Dungeon Overlay"),
    ]

    found = find_window("Darkest Dungeon", windows)

    assert found.hwnd == 22


def test_no_match() -> None:
    windows = [make_window(title="Calculator")]

    with pytest.raises(WindowNotFoundError, match="No visible window matched"):
        find_window("Darkest Dungeon", windows)


def test_ambiguous_substring_match() -> None:
    windows = [
        make_window(hwnd=31, title="Darkest Dungeon"),
        make_window(hwnd=32, title="Darkest Dungeon Debug"),
    ]

    with pytest.raises(AmbiguousWindowError, match="Multiple windows match"):
        find_window("Darkest", windows)


def test_ambiguous_exact_titles_are_rejected() -> None:
    windows = [
        make_window(hwnd=41, title="Darkest Dungeon", left=0),
        make_window(hwnd=42, title="Darkest Dungeon", left=100),
    ]

    with pytest.raises(AmbiguousWindowError):
        find_window("darkest dungeon", windows)


def test_minimized_window_rejection() -> None:
    window = make_window(minimized=True)

    with pytest.raises(MinimizedWindowError, match="minimized"):
        assert_window_capturable(window)


def test_zero_size_window_rejection() -> None:
    window = make_window(width=0, height=0)

    with pytest.raises(InvalidWindowGeometryError, match="invalid"):
        assert_window_capturable(window)


def test_negative_size_window_rejection() -> None:
    window = make_window(width=-10, height=1080)

    with pytest.raises(InvalidWindowGeometryError):
        assert_window_capturable(window)


def test_list_visible_windows_skips_hidden_and_untitled(monkeypatch: pytest.MonkeyPatch) -> None:
    records = {
        1: {"title": "", "visible": True, "rect": (0, 0, 100, 100), "iconic": False},
        2: {
            "title": "Hidden Game",
            "visible": False,
            "rect": (0, 0, 100, 100),
            "iconic": False,
        },
        3: {
            "title": "Darkest Dungeon",
            "visible": True,
            "rect": (10, 20, 1930, 1100),
            "iconic": False,
        },
    }

    def fake_enum(callback: object, extra: object) -> None:
        for hwnd in records:
            callback(hwnd, extra)  # type: ignore[operator]

    monkeypatch.setattr("ddca.capture.windows.ensure_process_dpi_aware", lambda: None)
    monkeypatch.setattr("ddca.capture.windows.win32gui.EnumWindows", fake_enum)
    monkeypatch.setattr(
        "ddca.capture.windows.win32gui.IsWindowVisible",
        lambda hwnd: records[hwnd]["visible"],
    )
    monkeypatch.setattr(
        "ddca.capture.windows.win32gui.GetWindowText",
        lambda hwnd: records[hwnd]["title"],
    )
    monkeypatch.setattr(
        "ddca.capture.windows.win32gui.GetWindowRect",
        lambda hwnd: records[hwnd]["rect"],
    )
    monkeypatch.setattr(
        "ddca.capture.windows.win32gui.IsIconic",
        lambda hwnd: records[hwnd]["iconic"],
    )

    windows = list_visible_windows()

    assert len(windows) == 1
    assert windows[0].hwnd == 3
    assert windows[0].title == "Darkest Dungeon"
    assert windows[0].left == 10
    assert windows[0].top == 20
    assert windows[0].width == 1920
    assert windows[0].height == 1080
    assert windows[0].minimized is False
