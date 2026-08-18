# Darkest Dungeon Combat Advisor

Read-only combat assistant for **Darkest Dungeon 1** on Windows Steam.

The tool observes the visible game window, and later phases will extract combat state and rank legal player actions. **You must perform every action yourself.** This project is not a bot, macro, trainer, or automation tool.

## Scope

This application may only read pixels from the visible game window.

It does **not**:

- read or modify game memory
- inject code into the game
- modify game files
- intercept network traffic
- simulate keyboard or mouse input
- automatically perform recommended actions

## Current status: Phase 1

Phase 1 can:

- enumerate visible titled desktop windows
- locate the Darkest Dungeon window by title
- capture that window once
- save a timestamped PNG and JSON metadata under a Git-ignored output directory

Later phases (calibration, combat-state recognition, strategy, overlay, telemetry, and ML) are **not implemented yet**.

## Requirements

- Windows 10/11
- Python 3.11
- PowerShell
- Darkest Dungeon 1 running in English for live capture (not required for unit tests)

GPU acceleration and PyTorch are not used in Phase 1.

## Setup (PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If `py -3.11` is not registered, create the venv with any Python 3.11 interpreter instead of switching versions:

```powershell
& "<path-to-python-3.11>\python.exe" -m venv .venv
```

## Tests

```powershell
python -m pytest -q
```

Unit tests mock Win32 and screenshot APIs. Darkest Dungeon does not need to be installed or running.

## Commands

List visible titled windows:

```powershell
python -m ddca.capture.cli list
```

Capture one screenshot of the game window:

```powershell
python -m ddca.capture.cli capture --title "Darkest Dungeon" --output-dir data/screenshots
```

Successful captures print the absolute paths of the PNG and JSON files.

## Output location

Default files are written to `data/screenshots/`:

- `capture_YYYYMMDDTHHMMSSZ.png`
- `capture_YYYYMMDDTHHMMSSZ.json`

That directory is Git-ignored. Do not commit raw screenshots, copyrighted game assets, logs, or local datasets.

## Troubleshooting

**Game not found.** Start Darkest Dungeon so its window is visible, then run the `list` command and pass an exact `--title` from that list.

**Minimized window.** Restore the game window. This tool will not restore, focus, or capture a minimized window.

**Multiple matching titles.** Narrow `--title` until exactly one window matches. The tool refuses to guess when more than one window matches.

**Administrator privilege mismatch.** If the game runs as Administrator, run PowerShell as Administrator too (or run both unelevated). Windows may hide elevated windows from a non-elevated process.

**Display scaling.** Phase 1 requests per-monitor DPI awareness before reading coordinates. If a capture is offset, use a single display scale, avoid mixed-DPI setups, and retry from a new PowerShell session.

## Current limitations

- Windows only
- One-shot capture; no continuous capture loop
- No combat-state recognition, strategy, overlay, telemetry, or machine learning
- Window matching is by title only
- Capture uses the Win32 window rectangle; exclusive fullscreen or unusual DPI layouts may still misalign

Recommendations in later phases will be labeled as best estimated actions under the current model, not proven optimal moves.

## Roadmap

1. **Phase 1** — Repository foundation and Windows game capture (current)
2. **Phase 2** — Calibration and region-of-interest system
3. **Phase 3** — Structured combat-state domain model
4. **Phase 4** — Deterministic rule-based strategy baseline
5. **Phase 5** — Combat-state visual recognition
6. **Phase 6** — Real-time English overlay
7. **Phase 7** — Telemetry and dataset collection
8. **Phase 8** — ML ranking after a labeled dataset exists
9. **Phase 9** — Evaluation and portfolio presentation
