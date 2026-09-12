# KickTrack

## What this is
A tool that processes fixed-camera/elevated football match footage and provides:
- **Player-level analytics**: heatmaps, distance covered, speed, sprints, zone occupancy
- **Match-level analytics**: formations, possession, team shape, passing networks

Current focus: **Plan 1 (done)** — detect and track players in a video with live markers.
See `docs/PLAN.md` for the full milestone checklist and what's next.

## Tech stack
- **CV/Tracking**: Python, Ultralytics YOLO (`yolov8m.pt`), BoT-SORT tracker
- **Package management**: `uv` (not pip/venv directly) — see `pyproject.toml`, lockfile is `uv.lock`
- **Backend (scaffolded, not yet the active focus)**: FastAPI, `app/main.py`
- **Hardware target**: Apple Silicon (M3 Pro) — inference uses `device="mps"`

## Key architecture decisions (and why)

- **BoT-SORT, not ByteTrack**: the test footage (stock aerial/elevated clips) isn't a
  perfectly static camera — it pans/zooms subtly. ByteTrack matches purely on IoU, so
  camera motion caused massive ID churn (new ID almost every frame). BoT-SORT's GMC
  (camera motion compensation) fixes this. Config: `scripts/botsort_custom.yaml`
  (`track_buffer: 90`, up from default 30, to survive brief occlusion/missed detections).

- **`imgsz=1280`, not lower**: dropping to 640/960 measurably misses real players
  (verified visually, not just by benchmark numbers) — not an acceptable trade for an
  analysis tool where the point is tracking everyone. Kept full resolution instead.

- **Async inference thread, not frame-skipping alone**: detection+tracking is slower
  than the video's real-time frame budget on this hardware. Rather than block playback
  on every inference call (causes visible stutter) or drop frames from the *video*
  (loses real playback frames), a background thread (`InferenceWorker` in
  `scripts/track_players.py`) runs YOLO continuously on the latest available frame and
  drops stale ones. The main thread only reads/displays/paces — never blocks on AI —
  so playback stays smooth at the cost of markers lagging the true position by ~1
  inference cycle (imperceptible for player movement speeds).

- **`uv` package management, `pyproject.toml` pinned to macOS** (`tool.uv.environments`
  restricted to `sys_platform == 'darwin'`) — avoids cross-platform dependency
  resolution failures for a Mac-only dev project.

## Known gotchas
- **Safe-chain proxy** wraps `uv`/npm on this machine and can block package resolution
  with "minimum package age" errors, or throttle/break large downloads (e.g. model
  weights from GitHub releases). Use `--safe-chain-skip-minimum-package-age` if a
  `uv sync` fails claiming a package "doesn't exist" when it clearly does on PyPI.
- Model weight downloads (`.pt` files from GitHub releases) have been observed to be
  very slow/unstable on this network — budget extra time, don't assume a hang means
  something is broken.

## File layout
- `scripts/track_players.py` — the tracking tool (see module for class breakdown:
  `PlayerTracker`, `InferenceWorker`, `MarkerRenderer`, `FramePacer`)
- `scripts/botsort_custom.yaml` — tracker tuning
- `data/videos/` — test footage (`match_1.mp4`, `match_2.mp4`, `match_3.mp4`)
- `docs/PLAN.md` — milestone checklist, updated as work progresses
- `app/` — FastAPI skeleton (not yet the active focus)

## Working conventions for this project
- Update `docs/PLAN.md` checkboxes as steps complete; add new "Plan N" sections for
  new milestones rather than editing Plan 1 in place.
- Don't touch dependencies (`pyproject.toml`) without checking first — the user runs
  installs themselves.
- Verify performance/quality claims empirically (benchmark timings, visually inspect
  sample frames) rather than assuming a change helped.
