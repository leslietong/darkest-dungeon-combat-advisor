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

## Current status: Phase 3B-2

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

Phase 3B-2 is a provisional deterministic baseline that classifies the four skill slots and Move as available, disabled, or unknown from those crops. It does not identify skill names.

**Not implemented:** skill identity, legal targets, CombatState inference, strategy scoring, utility, estimated win probability, continuous capture, and the external companion window.

GPU acceleration and PyTorch are not used.

## Phase 3A and the Observation → CombatState boundary

Phase 3A stops at structured data. It does not fill those structures from pixels.

- **Observation** types (`Observation`, `HeroObservation`, `EnemyObservation`, `ActionObservation`) bind per-field `ObservedValue`s to a `FrameReference` and optional `RegionEvidence`. They may name an ROI; they never store screenshot arrays.
- **State** types (`HeroState`, `EnemyState`, `ActionState`, `CombatState`) are the validated snapshot used by later phases. `HeroObservation.to_state()` (and the enemy/action equivalents) copy observed fields into state objects and drop evidence. `CombatState.from_observations(...)` assembles a party-level snapshot and enforces rank, identity, and action-slot invariants.
- Phase 3B-2 can populate `ActionObservation.is_available` from the five action-slot crops. It leaves `skill_id` and `legal_target_ranks` unknown (Move `skill_id` is not applicable). Filling the rest of CombatState from pixels is not implemented. Strategy ranking and estimated win probability belong in a later result model, not on `CombatState`. An external companion window is a future direction only; it is not a current capability.

```
Phase 1 capture (PNG + metadata)
    -> Phase 2 calibration ROIs (1111x654 v3, manually reviewed and accepted)
    -> Phase 3B-1 static ROI crops + evidence manifest (pixels only)
    -> Phase 3B-2 action-slot available/disabled/unknown (provisional)
    -> [remaining Phase 3B recognition: not implemented]
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

## Phase 3B-2: action-slot availability

`python -m ddca.vision.action_state_cli` reads a Phase 3B-1 extraction directory and classifies `skill_slot_1`–`skill_slot_4` plus `move_action_slot`.

Semantics:

- **available** — `ObservedValue(value=True, status=observed, source=classifier)`
- **disabled** — `ObservedValue(value=False, status=observed, source=classifier)`
- **unknown** — `ObservedValue(value=None, status=unknown, source=classifier)`

Unknown is never encoded as `False`. Disabled is never encoded as missing. The classifier uses inner-content chroma features (foreground ratio, brightness std, saturation, chromatic-pixel ratio) and keeps an abstention band between conservative disabled and available thresholds. Exact boundary values abstain to unknown because they have zero decision margin. A coloured border cannot make a grey inner icon available. A blank or nearly black crop is unknown, not disabled.

Confidence uses a class-side normalized margin: disabled distance is divided by `disabled_max_chroma`, available distance by `1 - available_min_chroma`. Those independent [0, 1] margins then pass through the same saturating transform and an input-quality factor, capped at 0.85. This is provisional classifier confidence, not a calibrated probability. Each classification records `crop_sha256` from the exact crop PNG bytes. The Phase 3B-1 manifest schema is unchanged and still does not store per-crop hashes.

```powershell
python -m ddca.vision.action_state_cli --extraction-dir output/extractions/example --config configs/vision/action_slot_state_v1.yaml --output output/action_state/example.json
```

Skill identity remains unknown. Legal targets are not computed. `CombatState` is not created.

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
- Phase 3B-2 is a provisional chroma baseline for action-slot available/disabled/unknown only; skill identity is not recognized
- CombatState inference from pixels remains unimplemented
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
4. **Phase 3B-1** — Static screenshot ROI extraction and evidence manifest (completed)
5. **Phase 3B-2** — Provisional action-slot available/disabled/unknown baseline (current)
6. **Phase 3B** — Remaining combat-state visual recognition (unimplemented)
7. **Phase 4** — Deterministic rule-based strategy baseline and estimated win probability (unimplemented)
8. **Phase 5** — Remaining recognition and robustness work
9. **Phase 6** — Real-time English overlay / external companion window (future direction; not implemented)
10. **Phase 7** — Telemetry and dataset collection
11. **Phase 8** — ML ranking after a labeled dataset exists
12. **Phase 9** — Evaluation and portfolio presentation
