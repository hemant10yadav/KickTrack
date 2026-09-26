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
- **CV/Tracking**: Python, Ultralytics YOLO (`yolo26s.mlpackage`, CoreML), BoT-SORT tracker
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
  inference cycle (imperceptible for player movement speeds). The worker does
  *only* inference + box extraction: turning raw boxes into players
  (`IdentityResolver` in `scripts/pipeline.py`: split suppression, identity,
  jersey sampling, confirm/grace state) runs on the display thread once per new
  result, because on a 50fps clip inference alone fills the 20ms budget and even
  ~0.7ms of extra worker-side work measurably raised the frame-drop rate
  (`docs/PLAN.md` Plan 2.8), while the display thread idles ~13ms per frame.

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

- **Rectangular `imgsz` (640x1152), not square (1280x1280)**: the actual footage is
  16:9, so a square export pads ~30% of the input with wasted letterbox pixels.
  Exporting at a rectangular size matching the footage aspect ratio drops only
  that padding, not player detail, so on-pitch detection recall is unchanged —
  verified visually frame-by-frame against the square export on both a 4K and a
  1080p clip; the only box-count difference was crowd/stand false positives,
  never a missed player. Went through 736x1280 first (inference 50.2ms ->
  34.8ms avg on `match_4.mp4`), then shrunk further to 640x1152 once the same
  visual-recall check passed again (34.8ms -> ~29-31ms avg) — this is what gets
  a 50fps source (`match_5.mp4`) under its 20ms native frame budget for the
  first time; worker frame drops on `match_5.mp4` went 1126/1750 (square@1280,
  64%) -> 795/1750 (rect@1280, 45%) -> 585/1750 (rect@1152, 33%). `match_3`/
  `match_4` (25-30fps) now drop 0 frames. `scripts/track_players.py`'s
  `INFERENCE_IMGSZ` must stay in sync with whatever shape the model was last
  exported at. Export command:
  `model.export(format="coreml", imgsz=(640, 1152), half=True)`.

- **YOLO26s, not yolov8s/v8m** (`docs/PLAN.md` Plan 2.9): every body swap in
  Plan 2.8 started with one detection box over two overlapping players. Measured
  on the three match_5 merge episodes, v8s often *does* return the second body,
  but at a confidence below BoT-SORT's `new_track_thresh` (0.5), so no track ever
  starts; lowering that threshold trades merges for churn (raw ids 63 -> 144),
  and v8m separates no better. YOLO26s (NMS-free head, same 640x1152 export)
  keeps two tracked boxes on 19/19 keeper+defender frames (v8s 3/19), 34/34
  blue+white frames (29/34) and 24/73 of the hardest pair (1/73), mints fewer
  ids, passes every hand-checked identity probe, and has equal visual recall on
  spot frames of match_4/match_5, for ~+0.4ms per cycle. It also returns a
  *container* box around two overlapping players alongside their own boxes on
  ~2% of frames; `SplitDetectionSuppressor` drops those. Export needs the numpy
  version ultralytics pins (coremltools 9.0 breaks under the project's numpy
  2.5), without touching the project env:
  `uv run --with "numpy==2.3.5" python -c "from ultralytics import YOLO; YOLO('yolo26s.pt').export(format='coreml', imgsz=(640, 1152), half=True)"`
  (`yolo26s.pt` comes from the `v8.4.0` release of `ultralytics/assets`; use
  `aria2c` on this network, the built-in downloader is very slow).
  (Int8 quantization would cut latency further but needs a new `scikit-learn`
  dependency for ultralytics' CoreML k-means quantization path — not added
  without checking first, see "Working conventions" below.)

- **`PlayerIdentityManager`, not raw BoT-SORT `track_id`, as the application's
  player identity**: debugged (while checking whether `match_5.mp4`'s frame
  drops were a quality regression) that BoT-SORT reassigns a brand-new
  `track_id` to the same physical player after a single missed frame — a
  small/distant, borderline-confidence box (median first-appearance area
  ~717px² on 3840x2160 footage) fails BoT-SORT's own IoU/motion match well
  before `track_buffer` (90 frames) would expire it. Confirmed by dumping
  BoT-SORT's internal tracked/lost pools frame-by-frame around one exact case
  (track 95 -> 159, ~9px apart, 1-frame gap). This is pre-existing — present
  equally on clean 25fps footage with no async frame-dropping at all — not
  caused by the imgsz work above. Fixed at two levels (see `docs/PLAN.md` Plan
  2.5 for the full numbers): `botsort_custom.yaml` thresholds tuned to reduce
  churn at the source, plus `PlayerIdentityManager` in `scripts/player.py` as
  an app-level safety net that reconciles a freshly-appearing `track_id`
  against recently-lost players (predicted position + bbox-size gate) before
  anything downstream (team classification, display confirm/grace state) keys
  off it. Ruled out: async frame-dropping itself (a synchronous, drop-free feed
  churned *worse*), GMC (only a partial contributor), appearance ReID (only a
  partial fix, added cost and an auto-downloaded model).

- **Identity is anchored to jersey color, because BoT-SORT swaps bodies without
  ever losing a track** (`docs/PLAN.md` Plan 2.8): the swaps seen on `match_5`
  (goalkeeper's id walking off with a defender, a blue player's id ending on a
  white one) all came from the detector returning one box over two bodies —
  BoT-SORT keeps its `track_id` on that box and it shrinks back onto the *other*
  body. No lost track, so reconciliation never runs. `PlayerIdentityManager`
  therefore carries an appearance per player, judges every clean single-body
  sighting against it with a *relative* test (far from its own player *and*
  clearly nearer some other known player's jersey — absolute thresholds fail on
  noisier footage like `match_4`), and puts a suspect live track back into the
  same joint assignment as the unmapped ones. An established owner only gives
  its track up when another row of that assignment takes the owner (its old
  body detected on its own) — acting on a merged box's blended color earlier
  than that was measured to oscillate and to lock in a wrong swap. Verify any
  change here three ways: the drop-free replay test
  (`tests/test_identity_replay.py`, recorded fixture, fast), the realtime CLI
  (paced, drops frames — a fix that only holds in `--output` mode is not a fix),
  and frame strips around every logged correction (`swap_log`), never counters
  alone. A per-frame dump of raw boxes + colors is the fastest way to iterate:
  the whole identity layer can be re-run on it in seconds without YOLO.

- **Markers are placed on the frame being shown, from a short result history**
  (`ResultTimeline` in `scripts/display.py`): the display holds each frame back
  `--display-delay-ms` (default 100ms, `PlaybackDelay`) so AI results usually exist
  on both sides of it and its boxes are interpolated onto that exact frame; a frame
  newer than every result gets them extrapolated from the two newest, so markers
  never freeze between inference cycles. Both run in video time (frame ids), not
  wall-clock time. `StalenessTracker` and `WorkerStats`/`DisplayStats` instrument
  exactly where time goes (inference vs. box-extraction vs. jersey-color sampling
  vs. display-loop segments) — always measure before optimizing here; two prior
  "obvious" fixes (jersey-extraction throttling, a display busy-wait spin) turned
  out to be non-issues or unnecessary once actually measured.

- **Frames are shrunk to `--display-width` (1920) right after decode**
  (`FrameScaler`, `docs/PLAN.md` Plan 2.10): on a 4K source `cv2.waitKey` repaints the
  window in ~30ms, over a 50fps clip's whole 20ms budget, while the resize costs
  0.4ms and the detector sees 1152px wide either way. Only calibration keyframes get
  the source frame; their homography is rescaled into working pixels. `--output`
  goes through ffmpeg's `h264_videotoolbox` (`FfmpegOutput`) to a file or a live
  stream URL (rtmp/srt/udp/rtsp); cv2's `mp4v` writer took 10.6ms per 4K frame.

- **Pitch homography between keyframes is propagated from an anchor frame, with
  corners spread over a grid** (`HomographyPropagator`, `docs/PLAN.md` Plan 2.11):
  frame-to-frame chaining drifted (a pan's 1-3px per frame sits inside RANSAC's
  threshold, so static broadcast graphics pulled toward "no motion"), and the
  strongest corners of a broadcast frame are its graphics. A keyframe is applied
  on top of what propagation had for its frame, not restarted from it, and one
  far from propagation is held back until the next confirms it. Check any change
  here with a replay against full calibrations of every 10th frame, not by eye
  alone.

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
- `scripts/track_players.py` — thin CLI entrypoint (`parse_args`, `main`)
- `scripts/pipeline.py` — video source resolution, raw YOLO/BoT-SORT detection,
  the background `InferenceWorker`, the display-thread `IdentityResolver` that
  turns its raw boxes into identified players, and the top-level `PlayerTracker`
  orchestrator
- `scripts/player.py` — everything "what is a player": `TeamClassifier` (jersey
  color → team), `StateManager` (confirm/grace visibility), `PlayerIdentityManager`
  (persistent `player_id` across BoT-SORT `track_id` churn *and* across the body
  swaps BoT-SORT makes without losing a track), `SplitDetectionSuppressor`
- `tests/fixtures/match_5_tracks.jsonl.gz` — recorded raw boxes + jersey colors
  for every frame of `match_5.mp4`; `tests/test_identity_replay.py` replays it
  through the identity layer in ~2s with no model or video (the fast way to
  check any identity change against real footage before a realtime run)
- `scripts/analytics.py` — everything "what do we measure from where players
  were": `PitchProjector` (pixels → pitch metres through the inverse homography),
  `PlayerTrace` (distance / speed from 0.5s bucket means, heat), `MatchAnalytics`
  (per-result sampling with occlusion/calibration gates, common-mode shift
  cancelling, `stats.json` + heatmap PNGs via `--analytics-dir`). See
  `docs/PLAN.md` Plan 3.1 for why per-frame steps are never summed and why
  everyone-moves-together is a calibration event, not motion.
- `scripts/ball.py` — everything "where is the ball and who has it":
  `BallTracker` (one physically plausible track from noisy candidates),
  `PossessionTracker`, `PassCounter` (team totals: completed / lost, short /
  long; passes settle 2s late so the live count only goes up), `TeamHistory`
  (which team a player was on around a pass), `BallAnalytics` (per-result
  driver, keeper sides, `passes.json`). See `docs/PLAN.md` Plan 3.2 for why the
  ball is captured from the raw detections by a predictor callback and why the
  tracker is not a Kalman filter, and Plan 3.3 for why a pass's team is never
  the team read at that instant. Verify pass changes against the match_5 hand
  labels (`tests/test_pass_replay.py`) *and* fresh realtime runs of both clips.
- `scripts/display.py` — everything "what gets shown on screen": `MarkerRenderer`
  (pins, ID labels, running-distance captions), `PitchMinimap` (live top-down
  positions and the ball), `BallRenderer` (ball ring, holder ring, pass panel),
  `PitchOverlayRenderer`,
  `ResultTimeline`, `PlaybackDelay`, `DisplaySmoother`, `FadeController`, `FramePacer`,
  `StalenessTracker`/`WorkerStats`/`DisplayStats`
- `scripts/botsort_custom.yaml` — tracker tuning
- `yolo26s.mlpackage` — the CoreML-exported model actually used at runtime (gitignored,
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
- **Group files by domain/concept, not one class per file.** When splitting up code,
  ask "what is this about" (player state/identity? team color? what's drawn on
  screen? the detection/tracking pipeline?) rather than pulling every class into
  its own file — a pile of one-class files is just as hard to navigate as one giant
  file, in the other direction. `scripts/player.py` and `scripts/display.py` are the
  reference examples: each holds several classes that all answer the same "what is
  this about" question, so a contributor can find where something lives by domain
  instead of memorizing a class-to-file map.
