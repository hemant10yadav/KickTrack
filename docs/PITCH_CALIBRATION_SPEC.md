# Pitch Calibration Spec (Plan 3 prerequisite)

## Goal

Map each tracked player's pixel position to real-world pitch coordinates
(meters, standard 105x68m pitch), so downstream Plan 3 analytics (distance
covered, speed, heatmaps, possession) can operate in physical units instead
of raw pixels.

## Constraints (established during scoping, 2026-09-12)

- **Camera source is not fixed-wide.** It follows the ball and regularly
  frames only half the pitch or less — this is the common case, not an
  edge case. Zoom level and angle are unknown in advance and can vary
  per source video.
- **Camera motion is continuous (pan/tilt/zoom following play), not hard
  cuts** — unlike broadcast footage that switches between fixed camera
  angles. The homography changes gradually, frame to frame.
- **Sometimes near-zero pitch markings will be visible** (e.g. mid-pitch
  pan with no nearby lines) — there will be windows with nothing usable
  to calibrate against.
- **Hardware budget is tight.** Detection/tracking already consumes most
  of the per-frame budget on the Neural Engine (~29-31ms/frame at
  640x1152, see `docs/PLAN.md` Plan 2.5). A per-frame heavyweight
  calibration model on top of that is not viable.
- Don't add new dependencies without checking with the user first (see
  `CLAUDE.md` "Working conventions").

## Why point-only homography (naive approach) doesn't work here

A single-image homography from 4+ point correspondences (e.g. pitch
corners, center circle) needs enough *spread-out* points visible in
frame. Half-pitch or tighter framing often won't have that — you might
see one penalty box and a strip of touchline, nothing else. Point-only
methods degrade badly or fail outright in that regime.

## Chosen approach: points + lines calibration, keyframe + propagation

This mirrors research built for exactly this problem (broadcast football,
which also reframes constantly): **PnLCalib** and **TVCalib** both fuse
point keypoints *and* detected line segments (including partial lines cut
off by the frame edge) into the homography solve, which degrades far more
gracefully under partial pitch visibility than point-only DLT.

Architecture, in three layers:

1. **Keyframe calibration**: run a points+lines model (start with
   PnLCalib's released pretrained weights, trained on the SoccerNet
   calibration dataset) on a subsampled set of frames, not every frame.
   Produces a homography + a confidence signal (inlier count / reprojection
   error).
2. **Frame-to-frame propagation**: between keyframes, update the
   homography cheaply (sparse optical flow or feature matching +
   incremental transform update) rather than re-running the full model —
   same "expensive AI once, cheap tracking fills the gaps" shape as
   `InferenceWorker`/`MotionExtrapolator` in `scripts/pipeline.py` /
   `scripts/display.py`. Re-trigger full keyframe detection when
   propagation confidence drops (too few inlier features, reprojection
   error over threshold, or a max-frame-age cap).
3. **Low-confidence fallback**: when a frame has too few pitch markings
   to calibrate at all (full keyframe detection *and* propagation both
   fail), hold the last-known homography and mark that frame's derived
   analytics as low-confidence rather than producing silently-wrong
   coordinates or crashing.

## Open questions

- ~~Does PnLCalib's pretrained model transfer to this project's
  elevated/fixed-rig camera angle at all?~~ **Answered by Phase 0: no** —
  zero detections on stand-camera footage (`match_3`), see Phase 0 results
  below. Fine-tuning or a different model is required before Phase 1.
- How often, in practice, do keyframes need to be re-detected for a
  panning/zooming clip — i.e. how fast does propagation confidence decay?
- What's the right inlier-count / reprojection-error threshold for
  "low confidence, hold last homography" vs. "trigger keyframe re-detect"?
  Needs empirical numbers from real clips, not a guess (per `CLAUDE.md`
  "verify performance/quality claims empirically").

## Phased delivery

### Phase 0 — Feasibility prototype (no pipeline integration yet) — DONE 2026-09-12

Goal: decide, with real numbers, whether PnLCalib-as-is is viable before
investing in the keyframe/propagation architecture.

**Setup:** PnLCalib (`mguti97/PnLCalib`, points+lines, SoccerNet-pretrained
`SV_kp`/`SV_lines` weights) cloned and run in-repo (`torch`/`torchvision`
already present transitively via `ultralytics`; `scipy`, `shapely`,
`lsq-ellipse` added under a new `pnlcalib-prototype` uv dependency group,
kept separate from the project's CoreML production deps). Ran
`inference.py --pnl_refine --device cpu` on 3 sampled frames each from
`match_3.mp4`, `match_4.mp4`, `match_5.mp4`, plus a synthetic half-pitch
crop of a `match_4` frame.

**Results:**

| Clip | Style | Result |
|---|---|---|
| `match_4` (3 frames + half-pitch crop) | Broadcast TV feed (Wembley) | Homography visually accurate on all 4 — reprojected lines matched real pitch markings tightly, including on the half-pitch crop |
| `match_5` (3 frames) | Broadcast TV feed (Etihad) | Same — accurate on all 3 |
| `match_3` (3 frames) | Elevated stand-camera, dim/grainy night footage | **Zero lines detected on any of the 3 frames** — complete failure, not partial degradation |

**Conclusion:** the half-pitch/zoom-framing concern that motivated the
points+lines architecture choice turned out *not* to be the binding
constraint — the half-pitch crop calibrated fine. The real constraint is a
**domain gap** between SoccerNet's broadcast-TV training footage and
`match_3`'s elevated stand-camera style.

Debugged further (raw heatmap inspection, `diagnose.py`) to find the exact
mechanism, since "model finds nothing" turned out to be imprecise:
- The model's raw heatmaps DO show peaks on `match_3` frames — max
  keypoint-channel score ~0.17-0.38 (vs. ~0.79 on `match_4`), max
  line-channel score ~0.46-0.69 (vs. ~0.87 on `match_4`) — all below the
  default thresholds (`kp_threshold=0.3434`, `line_threshold=0.7867`).
- **Hypothesis tested and ruled out:** "it's just an overly strict
  confidence threshold tuned for broadcast footage." Lowering both
  thresholds (0.15 / 0.3) does let `heuristic_voting` produce a
  homography on some frames — but the reprojected lines are visibly
  wrong (crossing at random angles, no alignment to real pitch markings).
- **Actual root cause:** the model's keypoint/line *localization* itself
  is degraded on this camera domain, not just its confidence — lowering
  the threshold surfaces noisy/incorrect correspondences, not correct
  ones the threshold was hiding. Likely contributing factors (not yet
  isolated): `match_3` is 2560x1440 vs. `match_4`'s 1920x1080, so the
  fixed `transform2 = T.Resize((540, 960))` resize path downsamples it
  differently; `match_3`'s night/floodlit color and contrast profile
  differs from typical broadcast color grading; and/or its camera
  elevation angle sits outside SoccerNet's training distribution.

This is a harder problem than "degrades gracefully under sparse
keypoints" (the case the points+lines architecture was chosen to
handle) — threshold-tuning is not a viable fix, confirmed empirically.

**Scope decision (2026-09-12):** `match_3`-style stand-camera footage is
out of scope for now — `match_4`/`match_5`-style broadcast footage is the
real target case KickTrack is being built against. Since PnLCalib
calibrates that footage reliably zero-shot (see table below), Phase 0's
exit criteria are met and no fine-tuning/model-swap detour is needed
before proceeding to Phase 1. Revisit `match_3`-style generalization
later if/when non-broadcast footage becomes an actual target again.

**Confidence/reliability data across all 7 broadcast-style test frames**
(3x `match_4`, 3x `match_5`, 1x `match_4` half-pitch crop) — all 7
produced a correct homography, with no degradation across frames, camera,
timestamp, or half-pitch framing:

| Frame | Top kp score | kp above threshold | Top line score | Lines above threshold |
|---|---|---|---|---|
| match_4_f0 | 0.76 | 17 | 0.86 | 4 |
| match_4_f1 | 0.79 | 27 | 0.87 | 12 |
| match_4_f2 | 0.76 | 25 | 0.87 | 5 |
| match_4_halfpitch_crop | 0.83 | 19 | 0.86 | 14 |
| match_5_f0 | 0.74 | 22 | 0.86 | 5 |
| match_5_f1 | 0.77 | 17 | 0.82 | 6 |
| match_5_f2 | 0.76 | 12 | 0.83 | 2 |

(For contrast, `match_3` frames scored 0.17-0.38 top keypoint confidence,
0/frames above threshold — see above. Out of scope per this decision.)

### Phase 1 — Keyframe calibration integration — DONE 2026-09-12

Wired keyframe-rate points+lines calibration into the tracking pipeline.
Implementation plan: `docs/superpowers/plans/2026-09-12-pitch-calibration-phase1.md`.

- Vendored PnLCalib's inference-only code into `scripts/pnlcalib/`
  (`model/`, `utils/`, `config/` subset — no training/eval code).
- `scripts/calibration.py`: `PitchCalibrator` (single-frame
  `calibrate(frame) -> CalibrationResult`, homography + confidence
  metrics) and `CalibrationWorker` (background thread, same
  submit/get_result pattern as `InferenceWorker`, calibrates only at a
  keyframe cadence since inference is 0.3-1.8s/frame — measured MPS
  ~0.33-0.41s, CPU ~1.6-1.8s on this project's M3 Pro).
- Wired into `PlayerTracker` (`scripts/pipeline.py`) alongside the
  existing `InferenceWorker`; verified end-to-end against
  `tests/videos/match_4.mp4` (headless run): pipeline completes normally,
  tracking stats unaffected, and the summary now reports
  `Calibration: homography_found=True keypoints=20 lines=1`.
- No consumer of the homography yet — that's Phase 3's `PlayerState`
  wiring below.

### Phase 2 — Frame-to-frame propagation

**Drift measurement (2026-09-13)**, before building anything: ran
`PitchCalibrator` on every frame of a 90-frame window (no propagation,
no holding stale) on `data/videos/match_4.mp4` and `match_5.mp4`, and
measured how far a fixed on-screen world point (pitch center) projects
frame-to-frame — i.e. how wrong a stale homography held for K frames
would be.

First attempt measured all 4 pitch corners too and got misleadingly huge
numbers (mean 196px, max 1130px at K=30) — traced to a measurement bug,
not a real problem: pitch corners are usually off-screen in these
zoomed/half-pitch shots, and homography extrapolation to points far
outside the observed keypoint region is numerically unstable even when
the on-screen calibration is fine (confirmed visually: frames 1565/1566
of match_4, the worst "corner" outlier pair, calibrate near-identically
on screen). Corrected to on-screen center point only:

| K (frames) | match_4 mean/median/max (px) | match_5 mean/median/max (px) |
|---|---|---|
| 1  | 9.5 / 6.4 / 57.6    | 5.8 / 1.3 / 42.5  |
| 5  | 21.8 / 18.6 / 60.7  | 9.2 / 5.8 / 43.0  |
| 10 | 40.0 / 34.2 / 87.9  | 13.0 / 9.2 / 45.1 |
| 15 | 59.9 / 52.3 / 121.6 | 14.9 / 12.2 / 47.2|
| 30 (current keyframe_interval) | 124.8 / 124.5 / 186.5 | 18.9 / 15.1 / 59.2 |

Mean and median are close together at every K (no heavy-tailed
instability) and drift grows roughly linearly with K — consistent with
real, gradual camera pan, not per-frame calibration noise.
**Conclusion:** `match_4`-style footage (more camera movement) drifts a
real, meaningful amount (~125px mean, ~6-10% of frame width) by the
current 30-frame keyframe interval — worth fixing. `match_5` drifts much
less in this window. Decision: build full propagation (not just shrink
`keyframe_interval`) per user direction — the throughput ceiling on
`keyframe_interval` (calibration takes ~0.35s/frame on MPS, so an
interval much below ~10 frames risks the worker falling behind its own
cadence at 25fps) means propagation is the right long-term fix, not a
workaround.

**Design:** `HomographyPropagator` (new class in `scripts/calibration.py`)
runs in `PlayerTracker`'s main thread (not the background worker) —
every displayed frame, not just at keyframe rate, mirroring how
`MotionExtrapolator` already runs per-frame in the main thread on top of
`InferenceWorker`'s keyframe-rate boxes (`scripts/display.py`). It tracks
sparse features (`cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`)
from the last full calibration's frame forward, composes the incremental
image-to-image transform with that keyframe's homography, and reports
propagation failure (too few tracked inliers) so the caller can fall back
to holding the last good homography rather than trusting a bad estimate —
the hold-on-failure behavior Phase 3 was going to add anyway, pulled
forward since propagation needs it too.

### Phase 3 — Low-confidence fallback + `PlayerState` integration

Add the hold-last-homography fallback and confidence flag, then wire
pixel→pitch coordinate conversion into the `PlayerState` work from
`docs/PLAN.md` Plan 3.

Phases 1-3 will get full bite-sized TDD implementation plans (via the
`writing-plans` skill) once Phase 0's prototype results are in — no
point locking in file-level task detail before we know if the model
choice itself holds up.
