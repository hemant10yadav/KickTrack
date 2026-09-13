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
  - [X] Rectangular `imgsz` matching the footage's 16:9 aspect ratio, instead of
        padding to a square — cuts wasted padding pixels, not player detail, so
        recall on the pitch is unchanged (verified visually on both a 4K and a
        1080p clip: same players/confidences detected at every step below; any
        box-count difference was crowd/stand false positives, never a missed
        player). Iterated twice:
          - 736x1280 first: inference 50.2ms -> 34.8ms avg on match_4.mp4.
          - Then 640x1152 (smaller still, same aspect): 34.8ms -> ~29-31ms avg,
            which gets a 50fps source (match_5.mp4) under its 20ms native frame
            budget for the first time — previously inference was *always* slower
            than 50fps playback, so the async worker was structurally guaranteed
            to drop frames no matter how it was tuned.
          - Frame drops on match_5.mp4 (full clip, 1750 frames): square@1280
            1126 dropped (64%) -> rect@1280 795 (45%) -> rect@1152 585 (33%).
            match_3/match_4 (25-30fps) dropped 0 frames at rect@1152.
          - Int8 quantization (a further latency lever) needs a new `scikit-learn`
            dependency ultralytics' CoreML export path pulls in for k-means
            quantization — not added since dependencies need checking with the
            user first (see CLAUDE.md).
          - Export command: `model.export(format="coreml", imgsz=(640, 1152), half=True)`.
  - [X] Debugged BoT-SORT track_id churn (found while checking match_5.mp4's frame
        drops didn't look like a quality regression — turned out to be pre-existing,
        present equally on clean 25fps footage). Root cause: a small/distant,
        borderline-confidence box (median first-appearance area ~717px^2 on
        3840x2160 footage) loses its IoU/motion match for a single missed frame and
        gets a brand-new track_id, well before `track_buffer` (90 frames) would
        expire it — confirmed by dumping BoT-SORT's internal tracked/lost pools
        frame-by-frame around one exact case (track 95 -> 159, ~9px apart, 1-frame
        gap). Fixed two levels up:
          - `botsort_custom.yaml`: raised `new_track_thresh` 0.25->0.5 (a fresh
            track now needs real confidence to exist) and `match_thresh` 0.8->0.9
            (more permissive re-matching of marginal-confidence redetections to
            their old/lost track). Cut raw BoT-SORT track_id churn on match_4.mp4
            (300 frames, ~22-28 real people) from 52 -> 29 unique IDs.
          - `PlayerIdentityManager` (new, in `track_players.py`): an app-level
            identity layer above BoT-SORT's own track_id, so whatever churn the
            tracker-level tuning doesn't catch is absorbed before it reaches team
            classification / display state. Reconciles a freshly-appearing
            track_id against recently-lost players using predicted position
            (velocity-extrapolated over the frame gap) normalized by bbox height,
            plus a size-ratio gate — deliberately conservative (start strict on
            merges, loosen only against measured false-splits).
          - Ruled out along the way: frame drops from the async worker (a
            synchronous, drop-free feed churned *worse*, not better); GMC/camera
            motion compensation (disabling it only cut churn 79->64, not the main
            cause); appearance ReID (`with_reid: True`, only cut ~64->50 IDs, added
            per-box cost and an auto-downloaded model — not adopted).
          - Tests: `tests/test_player_identity.py` (fast, synthetic — pins the
            exact 95->159 case plus the merge gates) and
            `tests/test_tracking_quality.py` (slow, real footage — churn-ratio and
            frame-drop-ratio regression ceilings on match_3/4/5).

- [X] **Plan 2.6: Player ID stability (measured on real footage, not synthetic tests)**

  Motivation: on-screen player numbers were visibly changing mid-clip on
  match_5.mp4, which would silently corrupt every per-player Plan 3 stat
  (distance, speed, heatmaps are all keyed by `player_id`). Each round below
  was found by running the real realtime CLI and dumping annotated frames
  (a `--output` run drops ~40% of frames and hid the first bug entirely);
  all numbers are from the realtime run of `match_5.mp4` (1750 frames, 0 skipped).

  - [X] Crossing-swap correction thrashed forever (33 swaps, all one pair,
        every 3 frames). `_swap()` flipped the tracker mapping but left
        `TeamClassifier`'s 0.7-smoothed color history on the wrong player_id,
        so `team_for()` kept reporting the wrong team and re-triggered an undo
        swap. Fix: move the color history with the identity; plus a
        `SWAP_COOLDOWN_FRAMES` backstop. 33 -> 1 swap.
  - [X] Greedy per-detection reconciliation was order-dependent. Replaced with
        a per-frame bipartite assignment (`scipy.optimize.linear_sum_assignment`)
        plus `AMBIGUITY_MARGIN`: a runner-up within 30% of the best refuses the
        merge and mints a new id (a wrong split is visible/recoverable; a wrong
        merge silently corrupts two players). Exposed and fixed two latent
        state bugs: a recycled BoT-SORT track_id clobbering another player's
        mapping, and a lost player whose own track reappeared being matched to
        a second detection in the same frame (`KeyError` on real footage).
  - [X] `MAX_NORM_DIST` 3.0 -> 1.5: every wrong merge in the frame-1436 cluster
        scored 1.87-2.86 (right under the old ceiling); every legitimate one
        scored < 1.0. Verified visually (ID 45 -> 3, ID 47 -> 48 renumbering
        gone) and against the slow regression ceilings (33/37/34 ids minted
        vs 70/45/65).
  - [X] Long-gap color match had no position or ambiguity gate and, after the
        above, carried ~90% of merges: it took the closest *color* anywhere on
        the pitch, and teammates all match. Now color is a gate, candidates
        must be reachable (`MAX_TRAVEL_PER_FRAME` * gap), are ranked by travel
        distance, and a close runner-up refuses. Also: only an *active*
        candidate may merge, one target per cycle (two tracks were merging
        into player 37 in consecutive frames -> duplicate ids on screen), and
        the target's stale tracker mapping is dropped on merge.
  - [X] `VELOCITY_HORIZON_FRAMES` = 10: constant-velocity extrapolation over
        25+ frame gaps landed far from where a player who slowed down actually
        reappeared (a 0.64-h reappearance went unmatched).
  - [X] Duplicate BoT-SORT tracks -- the dominant remaining cause. A
        same-position ID-change detector found 55 visible flips, most of them
        one pair alternating (45 <-> 3 eight times); frame dumps showed two
        labels stacked on one body (ID 3 + ID 22, ID 58 + ID 1). BoT-SORT was
        running two tracks on one player; both were already mapped, so no
        reconciliation path ever saw them. `_check_aliases`: two players whose
        boxes overlap (IoU > 0.55, similar size, not different teams) on 4
        frames in a 45-frame window are merged into the older id -- vetoed if
        the pair was seen clearly apart in that window (real players lining
        up in perspective are seen apart before/after; duplicate tracks never
        are). 55 -> 29 visible flips (many of the rest are the one-time snap
        to the kept id); 11 pairs aliased, spot-checked visually (14 into 10:
        both labels on one player at frames 857 and 1034).
  - Net on match_5.mp4: reconciliations 27 -> 6, swap corrections 33 -> 1.
    Tests: `tests/test_player_identity.py` 11 -> 17 (order-independence,
    ambiguity refusal, teleport-distance color match refused, two equidistant
    lost teammates refused, alias merge, seen-apart veto);
    `tests/test_tracking_quality.py` 6/6 with frame drops unchanged.
  - Known limit (from the literature -- GTA, SoccerNet GSR): two same-kit
    teammates shoulder-to-shoulder are indistinguishable by position + team
    color; SOTA sits at HOTA ~81-83%, not 100%. Next levers if needed:
    per-player appearance embeddings (not team-average color), and an offline
    global tracklet re-association pass for analytics-grade tracks.

- [ ] **Plan 3: Player & Match Analytics**

  - [ ] `PlayerState`: per-track position (pitch-relative, once calibration exists),
        velocity, team, distance covered, speed — built on top of the existing
        tracking + team classification pipeline
  - [ ] Heatmaps per player (and per team)
  - [ ] Possession detection (nearest player to ball, possession changes over time)
  - [ ] (Longer-term, not yet scoped in detail) formations/team shape, passing networks
