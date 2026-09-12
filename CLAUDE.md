# KickTrack

## What this is
A tool that processes fixed-camera/elevated football match footage and provides:
- **Player-level analytics**: heatmaps, distance covered, speed, sprints, zone occupancy
- **Match-level analytics**: formations, possession, team shape, passing networks

Current focus: **Plan 1 & 2 (done)** — player tracking with live markers, team
classification by jersey color, and the tracking pipeline is now latency-optimized.
**Plan 3 (active)** — building actual analytics (player state, heatmaps, possession)
on top of the tracking pipeline. See `docs/PLAN.md` for the full milestone checklist.

## Tech stack
- **CV/Tracking**: Python, Ultralytics YOLO (`yolov8m.mlpackage`, CoreML), BoT-SORT tracker
- **Package management**: `uv` (not pip/venv directly) — see `pyproject.toml`, lockfile is `uv.lock`
- **Backend (scaffolded, not yet the active focus)**: FastAPI, `app/main.py`
- **Hardware target**: Apple Silicon (M3 Pro) — inference runs on the Neural Engine via
  CoreML (`ComputeUnit.CPU_AND_NE`), not PyTorch/MPS (see below for why)
- **Python 3.13, not 3.14**: `coremltools` has no working native extensions on 3.14
  yet (source-built wheel loads but both export and even *loading* a `.mlpackage`
  fail with `BlobWriter not loaded` / `Unable to load libmodelpackage`). 3.13 has a
  prebuilt wheel and works cleanly. `pyproject.toml`'s `requires-python = ">=3.13"`
  already permitted this — no constraint change needed, just recreate `.venv` on 3.13
  if it drifts back to 3.14.

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

- **CoreML (`yolov8m.mlpackage`), not PyTorch/MPS**: measured and instrumented before
  choosing (see `docs/PLAN.md` Plan 2 history) — PyTorch/MPS inference at `imgsz=1280`
  averaged ~70ms/frame. Tried smaller/faster models and lower resolutions first
  (`yolov8s`, `imgsz=960/640`) but all of them measurably lost real player detections
  (verified visually, not just by box-count stats) — disqualifying for a tool whose
  point is tracking everyone. CoreML export of the *same* `yolov8m` model at the *same*
  `imgsz=1280` runs at ~57-61ms/frame on the Neural Engine with equal-or-better
  detection quality — a strict improvement, not a tradeoff. Export command:
  `model.export(format="coreml", imgsz=1280, half=True)` (must be run from a Python
  3.13 venv — see above).

- **Async inference worker keeps latest + previous result, not just latest**: enables
  `MotionExtrapolator` to estimate each track's velocity and shift its displayed
  position forward by however long it's been since the last AI result, so markers
  keep moving smoothly between inference cycles instead of freezing at a stale
  position. `StalenessTracker` and `WorkerStats`/`DisplayStats` (in
  `scripts/track_players.py`) instrument exactly where time goes (inference vs.
  box-extraction vs. jersey-color sampling vs. display-loop segments) — always
  measure before optimizing here; two prior "obvious" fixes (jersey-extraction
  throttling, a display busy-wait spin) turned out to be non-issues or unnecessary
  once actually measured.

## Known gotchas
- **Safe-chain proxy** wraps `uv`/npm on this machine and can block package resolution
  with "minimum package age" errors, or throttle/break large downloads (e.g. model
  weights from GitHub releases). Use `--safe-chain-skip-minimum-package-age` if a
  `uv sync` fails claiming a package "doesn't exist" when it clearly does on PyPI.
- Model weight downloads (`.pt` files from GitHub releases) have been observed to be
  very slow/unstable on this network — budget extra time, don't assume a hang means
  something is broken.
- **`coremltools` needs a matching Python version.** On Python 3.14 it silently
  builds from source (no prebuilt wheel) and produces a package with broken native
  extensions — both `model.export(format="coreml")` and loading a `.mlpackage` back
  fail (`BlobWriter not loaded`, `Unable to load libmodelpackage`). Stay on Python
  3.13, where a prebuilt wheel exists and everything works.

## File layout
- `scripts/track_players.py` — the tracking tool (see module for class breakdown:
  `PlayerTracker`, `InferenceWorker`, `MarkerRenderer`, `FramePacer`,
  `MotionExtrapolator`, `TeamClassifier`, `StalenessTracker`/`WorkerStats`/`DisplayStats`)
- `scripts/botsort_custom.yaml` — tracker tuning
- `yolov8m.mlpackage` — the CoreML-exported model actually used at runtime (gitignored,
  like other `.pt`/model weight files — regenerate with the export command above)
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
