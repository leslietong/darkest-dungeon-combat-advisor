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

## Current status: Phase 3B-1

The windowed **1111 x 654 v3** calibration was **manually reviewed and accepted**. Other resolutions, large enemies, tooltip states, status icons, and other battle states remain unverified. Do not change those coordinates unless a new resolution profile is added.

Phase 1 can:

- enumerate visible titled desktop windows
- locate the Darkest Dungeon window by title
- capture that window once
- save a timestamped PNG and JSON metadata under a Git-ignored output directory

Phase 2 adds:

- a captured-window normalized coordinate system (`0` to `1`)
- a YAML calibration profile for the manually reviewed and accepted windowed **1111 x 654 v3** layout; other resolutions, large enemies, tooltip states, status icons, and other battle states remain unverified
- named regions for the game frame, combat area, hero/enemy ranks 1-4, per-rank health bars, four skill slots plus Move, round counter, and current-hero panel
- YAML `validation_status: confirmed` or `provisional` so schema validity is distinct from visual confirmation
- a labeled debug preview overlaid on a captured screenshot
- a pixel-to-normalized helper for manual YAML edits

Phase 3A adds a pure-domain combat package (`ddca.combat`) that can represent a battle snapshot in memory and as deterministic JSON. It does not look at pixels.

Phase 3B-1 extracts configured ROIs from a static screenshot and writes lossless crops plus an evidence manifest. It performs no recognition or inference.

**Not implemented:** Phase 3B recognition, strategy scoring, estimated win probability, continuous capture, and the external companion window.

GPU acceleration and PyTorch are not used.

## Phase 3A and the Observation → CombatState boundary

Phase 3A stops at structured data. It does not fill those structures from pixels.

- **Observation** types (`Observation`, `HeroObservation`, `EnemyObservation`, `ActionObservation`) bind per-field `ObservedValue`s to a `FrameReference` and optional `RegionEvidence`. They may name an ROI; they never store screenshot arrays.
- **State** types (`HeroState`, `EnemyState`, `ActionState`, `CombatState`) are the validated snapshot used by later phases. `HeroObservation.to_state()` (and the enemy/action equivalents) copy observed fields into state objects and drop evidence. `CombatState.from_observations(...)` assembles a party-level snapshot and enforces rank, identity, and action-slot invariants.
- Recognition that would populate observations from a capture is **Phase 3B** and is not implemented. Strategy ranking and estimated win probability belong in a later result model, not on `CombatState`. An external companion window is a future direction only; it is not a current capability.

```
Phase 1 capture (PNG + metadata)
    -> Phase 2 calibration ROIs (1111x654 v3, manually reviewed and accepted)
    -> Phase 3B-1 static ROI crops + evidence manifest (pixels only)
    -> [Phase 3B recognition: not implemented]
    -> Observation / HeroObservation / EnemyObservation / ActionObservation
    -> CombatState (validated snapshot, per-field confidence)
    -> [strategy / estimated win probability: not implemented]
    -> [external companion window: future direction only]
```

## Phase 3B-1: static ROI extraction

`python -m ddca.vision.extraction_cli` crops every configured region (or a selected subset) from an existing screenshot. It writes PNG crops and `manifest.json`. It does not identify heroes, enemies, skills, HP, rounds, or statuses, and it does not build a `CombatState`.

The schema 1 manifest records source-image and calibration SHA-256 hashes, a stable `frame_id` equal to the source-image digest, optional `captured_at_utc` from a matching capture sidecar, and `extracted_at_utc`. It stores the source basename only. Each region includes the calibration `validation_status` (`confirmed` or `provisional`). Duplicate `--region` values are ignored after the first occurrence, preserving request order.

```powershell
python -m ddca.vision.extraction_cli --image data/screenshots/capture_YYYYMMDDTHHMMSSZ.png --profile configs/calibration/windowed_1111x654.yaml --output-dir output/extractions/example
```

Optional: `--region skill_slot_1` (repeatable) and `--overwrite` to replace only that extraction's crops and manifest.

`active_hero_marker` and enemy rank 3/4 health regions remain provisional. Status-icon ROIs remain undefined. Extraction only proves that a configured rectangle can be cropped.

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

**Wrong resolution.** Only the windowed 1111 x 654 v3 calibration was manually reviewed and accepted. Other sizes still map normalized coordinates, but those resolutions and other battle configurations remain unverified. Recapture at 1111 x 654 or add a new YAML profile.

## Current limitations

- Windows only
- One-shot capture; no continuous capture loop
- Only the windowed 1111 x 654 v3 calibration was manually reviewed and accepted; other resolutions, large enemies, tooltip states, status icons, and other battle states remain unverified
- `game_frame` and character ranks are confirmed on that accepted v3 profile; HUD boxes were fitted from one battle screenshot
- The action bar covers four selected skill slots plus Move; later strategy logic must treat Move as a legal combat action
- `active_hero_marker` remains provisional and still requires validation using a screenshot from another hero's turn
- Enemy ranks 3 and 4 health regions remain provisional because those positions are empty in the calibration screenshot
- Status-effect icon regions are uncalibrated and require a screenshot with visible Bleed, Blight, Stun, Buff, or Debuff
- Phase 3A is a domain model only: no OCR, no OpenCV recognition, no template matching, no classifiers, no Steam/game-file parsing, and no filling of CombatState from pixels
- Phase 3B-1 extracts configured ROIs and writes an evidence manifest; it does not recognize or infer game values
- Phase 3B visual recognition is not implemented
- Strategy scoring and estimated win probability are not implemented
- Continuous capture is not implemented
- The external companion window is not implemented; it is a future direction only
- Overlay, telemetry, and ML/DL ranking are not implemented
- Window matching is by title only
- Capture uses the Win32 window rectangle, including OS chrome; `game_frame` is the 16:9 client inside that chrome

Recommendations in later phases will be labeled as best estimated actions under the current model, not proven optimal moves.

## Roadmap

1. **Phase 1** — Repository foundation and Windows game capture
2. **Phase 2** — Calibration and region-of-interest system (1111 x 654 v3 manually reviewed and accepted; other layouts unverified)
3. **Phase 3A** — Structured combat observation and state models
4. **Phase 3B-1** — Static screenshot ROI extraction and evidence manifest (current)
5. **Phase 3B** — Combat-state visual recognition (unimplemented)
6. **Phase 4** — Deterministic rule-based strategy baseline and estimated win probability (unimplemented)
7. **Phase 5** — Remaining recognition and robustness work
8. **Phase 6** — Real-time English overlay / external companion window (future direction; not implemented)
9. **Phase 7** — Telemetry and dataset collection
10. **Phase 8** — ML ranking after a labeled dataset exists
11. **Phase 9** — Evaluation and portfolio presentation
