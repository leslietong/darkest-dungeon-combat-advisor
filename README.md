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

## Current status: Phase 2

Phase 1 can:

- enumerate visible titled desktop windows
- locate the Darkest Dungeon window by title
- capture that window once
- save a timestamped PNG and JSON metadata under a Git-ignored output directory

Phase 2 adds:

- a captured-window normalized coordinate system (`0` to `1`)
- a YAML calibration profile for one validated resolution: **1111 x 654** windowed
- named regions for the game frame, combat area, hero/enemy ranks 1-4, per-rank health bars, four skill slots plus Move, round counter, and current-hero panel
- YAML `validation_status: confirmed` or `provisional` so schema validity is distinct from visual confirmation
- a labeled debug preview overlaid on a captured screenshot
- a pixel-to-normalized helper for manual YAML edits

Later phases (combat-state recognition, strategy, overlay, telemetry, and ML) are **not implemented yet**.

GPU acceleration and PyTorch are not used in Phase 1 or Phase 2.

## Requirements

- Windows 10/11
- Python 3.11
- PowerShell
- Darkest Dungeon 1 running in English for live capture (not required for unit tests)

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

Validate the bundled calibration profile:

```powershell
python -m ddca.vision.cli validate --calibration configs/calibration/windowed_1111x654.yaml
```

Draw labeled ROI rectangles on a captured screenshot:

```powershell
python -m ddca.vision.cli preview --image data/screenshots/capture_YYYYMMDDTHHMMSSZ.png --calibration configs/calibration/windowed_1111x654.yaml --output output/battle_calibration_preview_v3.png --strict-size
```

The calibrated action bar includes the four currently selected hero skill slots plus the Move action slot. Move is a legal combat action that later decision logic must consider. Phase 2 only defines the ROI; it does not recognize icons or recommend actions.

Optional: draw one ROI group (`frame`, `ranks`, `health`, `actions`, `turn`, `containers`, `provisional`):

```powershell
python -m ddca.vision.cli preview --image data/screenshots/capture_YYYYMMDDTHHMMSSZ.png --calibration configs/calibration/windowed_1111x654.yaml --output output/action_bar_preview_v3.png --group actions --strict-size
```

Convert a pixel rectangle measured in Paint or similar into YAML unit coordinates:

```powershell
python -m ddca.vision.cli from-pixels --left 8 --top 30 --width 1095 --height 616 --image-width 1111 --image-height 654
```

Edit `configs/calibration/windowed_1111x654.yaml`, then regenerate the preview. Do not commit the preview PNG or raw game screenshots.

## Output location

Default files are written to `data/screenshots/`:

- `capture_YYYYMMDDTHHMMSSZ.png`
- `capture_YYYYMMDDTHHMMSSZ.json`

`data/screenshots/` is Git-ignored. Do not commit raw screenshots, copyrighted game assets, logs, or local datasets.

Labeled calibration previews default to `output/calibration_preview.png`. That directory is Git-ignored.

## Troubleshooting

**Game not found.** Start Darkest Dungeon so its window is visible, then run the `list` command and pass an exact `--title` from that list.

**Minimized window.** Restore the game window. This tool will not restore, focus, or capture a minimized window.

**Multiple matching titles.** Narrow `--title` until exactly one window matches. The tool refuses to guess when more than one window matches.

**Administrator privilege mismatch.** If the game runs as Administrator, run PowerShell as Administrator too (or run both unelevated). Windows may hide elevated windows from a non-elevated process.

**Display scaling.** Phase 1 requests per-monitor DPI awareness before reading coordinates. If a capture is offset, use a single display scale, avoid mixed-DPI setups, and retry from a new PowerShell session.

**ROI boxes look wrong.** Open a combat screenshot preview and compare each labeled box to the HUD. Measure pixel rectangles on the PNG, convert them with `from-pixels` (full-image pixels), then enter the values as `relative_to: game_frame` fractions, or as fractions of another parent region. `game_frame` should exclude the Windows title bar and border.

**Status icons.** The current battle screenshot does not show Bleed, Blight, Stun, Buff, or Debuff icons. Status-icon ROIs are therefore omitted and are not visually confirmed. Capture a second combat screenshot with visible status effects before adding them.

**Wrong resolution.** Phase 2 is validated at 1111 x 654 windowed. Other sizes still map normalized coordinates, but they are not treated as reliable yet. Recapture at 1111 x 654 or add a new YAML profile.

## Current limitations

- Windows only
- One-shot capture; no continuous capture loop
- Calibration currently validated for one windowed size: 1111 x 654
- `game_frame` and character ranks are confirmed; HUD boxes were fitted from one battle screenshot and still need human review of the v3 preview
- The action bar covers four selected skill slots plus Move; later strategy logic must treat Move as a legal combat action
- `active_hero_marker` still requires validation using a screenshot from another hero's turn
- Enemy ranks 3 and 4 health regions remain provisional because those positions are empty in the calibration screenshot
- Status-effect icon regions are uncalibrated and require a screenshot with visible Bleed, Blight, Stun, Buff, or Debuff
- Phase 3 has not started; no combat-state recognition, strategy, overlay, telemetry, or machine learning is implemented
- Window matching is by title only
- Capture uses the Win32 window rectangle, including OS chrome; `game_frame` is the 16:9 client inside that chrome

Recommendations in later phases will be labeled as best estimated actions under the current model, not proven optimal moves.

## Roadmap

1. **Phase 1** — Repository foundation and Windows game capture
2. **Phase 2** — Calibration and region-of-interest system (current)
3. **Phase 3** — Structured combat-state domain model
4. **Phase 4** — Deterministic rule-based strategy baseline
5. **Phase 5** — Combat-state visual recognition
6. **Phase 6** — Real-time English overlay
7. **Phase 7** — Telemetry and dataset collection
8. **Phase 8** — ML ranking after a labeled dataset exists
9. **Phase 9** — Evaluation and portfolio presentation
