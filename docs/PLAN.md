## Goal

Take a fixed-camera football video and show a marker on each player that moves with them as the video plays.

- [X] **Plan 1: Track Players in a Video**

  - [X] Add CV dependencies (Ultralytics YOLO, OpenCV) to requirements
  - [X] Install dependencies
  - [X] Sample fixed-camera football video to test with
  - [X] Script: read video frame by frame (OpenCV)
  - [X] Script: detect players in each frame (YOLO, person class)
  - [X] Script: track each player across frames with a consistent ID (ByteTrack)
  - [X] Script: draw a marker (dot/box + ID) on each tracked player
  - [X] Script: display the annotated video live as it processes (`cv2.imshow`)
  - [X] Run it end-to-end and confirm markers stay on the correct player as they move
- [X] **Plan 2: Team Classification (Jersey Colors)**

  - [X] Crop the jersey region from each tracked player's bounding box
  - [X] Extract a dominant color per player (e.g. k-means on jersey pixels)
  - [X] Cluster players into Team A / Team B / Referee based on dominant color
  - [X] Keep each track ID's team assignment stable across frames (don't flip-flop)
  - [X] Color-code markers per team (e.g. red vs blue, referee in yellow/black)
  - [X] Run end-to-end and confirm team assignment stays consistent as players move

- [X] **Plan 2.5: Tracking Pipeline Latency (measure, don't guess)**

  - [X] Fix camera-motion ID churn (BoT-SORT with GMC, replacing ByteTrack)
  - [X] Async inference worker (display never blocks on AI) — fixed choppy playback
  - [X] Instrument staleness (frames/ms behind) instead of assuming it's fine
  - [X] Motion extrapolation (velocity-based) so markers don't freeze between AI updates
  - [X] Instrument worker (inference vs. extract_boxes vs. jersey vs. residual) and
        display loop (read/submit/draw/imshow/wait) segment-by-segment
  - [X] Batch GPU→CPU tensor transfer in `extract_boxes` (was ~23ms/frame, now ~0.5ms)
  - [X] Deadline-based `FramePacer` (fixed a silent FPS drift from `waitKey` overshoot)
  - [X] Benchmark model/resolution matrix (v8m/v8s × 1280/960/640) — verified visually,
        not just by box-count stats, since a "sweet spot" model+resolution combo can
        look fine on paper while actually missing real players
  - [X] CoreML export (Neural Engine) — adopted `yolov8m.mlpackage` @ 1280: same/better
        detection quality as PyTorch/MPS, ~18% faster (57-61ms vs 70ms/frame)
  - [X] Migrate project to Python 3.13 (coremltools has no working native extensions
        on 3.14 yet)

- [ ] **Plan 3: Player & Match Analytics**

  - [X] `PlayerState` + `StateManager`: canonical per-track data (bbox, position
        (feet, pixels for now), velocity, team, confidence, last_seen) built on top of
        the existing tracking + team classification pipeline — replaces the raw
        `(x1, y1, x2, y2, track_id)` tuples that used to flow through
        `InferenceResult`/`MotionExtrapolator`; `MotionExtrapolator` now extrapolates
        directly from each `PlayerState`'s own smoothed velocity
  - [X] Occlusion grace period: a player YOLO misses for one cycle no longer vanishes
        instantly — `StateManager` keeps coasting on last known position/velocity for
        up to `COASTING_GRACE_SECONDS` (0.75s), evicts states unseen for 5s (bounds
        memory, avoids stale-ID reuse pollution), and marks coasting markers
        (`is_coasting`) so a guess renders visibly dimmer than a real detection
  - [ ] Pitch homography: standalone calibration script — click 4-6 known pitch
        landmarks (penalty box corners, center circle) on one frame, compute the
        transform, save the matrix. Converts pixel positions to real pitch
        coordinates, so speed/distance become actual m/s and meters, not pixel math
  - [ ] Heatmaps per player (and per team)
  - [ ] Possession detection (nearest player to ball, possession changes over time)
  - [ ] (Longer-term, not yet scoped in detail) formations/team shape, passing networks
