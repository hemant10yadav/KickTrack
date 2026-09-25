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

- [X] **Plan 2.6: Identity Swaps When Players Converge**

  Plan 2.5 fixed *fragmentation* (one player, a new `track_id` after a gap). The
  other failure mode is a *swap*: two players converge and the tracks come apart
  attached to the wrong bodies. It is the more damaging of the two for Plan 3 —
  fragmentation loses data and looks broken, a swap silently moves one player's
  distance and heatmap onto another's and looks fine. `PlayerIdentityManager` had
  no defense against it: reconciliation only ever considered *lost* players, and
  its greedy per-detection argmin with a 3.0-bbox-height radius would happily
  hand a detection its neighbour during a crossing.

  - [X] **Team-color gate.** Jersey color is already sampled for team
        classification, so refusing a merge between two different teams costs
        nothing and rules out roughly the half of crossings that are contested
        balls between opponents. `TeamClassifier.team_for_color` classifies a
        color sampled outside the per-track history; `TeamGate` (one instance per
        worker cycle, box samples cached) answers "what team is this player" and
        "what team is this not-yet-identified box". Only gates when both sides
        are known, so it can't block merges before the clusters are fit.
        Refused 7 merges on match_5.mp4.
  - [X] **Joint assignment, not greedy.** All of a cycle's unmapped detections
        are matched against all lost candidates in one `linear_sum_assignment`
        (scipy, already a dependency), gates expressed as an unreachable cost.
        Order-independent, and two detections can no longer compete for one
        player — the greedy version let the first detection claim a player that
        a later one fit far better, which is exactly what a crossing produces.
  - [X] **Occlusion-aware state.** While a box overlaps another detection, or has
        swallowed a track that vanished into it (the detector merging two players
        into one box — no second detection to overlap, so the tell is a player
        going lost inside it), the box is partly someone else's. Velocity freezes
        at its last clean value and matching predicts from the last clean
        position/frame, so players emerging from a merge are resolved by the
        motion they carried in rather than the merged box's meaningless drift. A
        box sitting on its own just-lost track is a reacquisition, not an
        occlusion, and is not flagged. `occluded_player_ids` exposes the
        contaminated players per cycle so Plan 3 stats can drop those samples
        instead of trusting a wrong position — 173 player-cycles on match_5.mp4.
  - [X] Tests: six scenarios in `tests/test_player_identity.py` (crossing with a
        merged box and new track_ids on both sides; cross-team merge refused, and
        the same geometry merged when teams match; an ordering that defeats greedy
        matching; velocity freeze; occluded-player reporting). `test_tracking_quality.py`
        churn ceilings unchanged on match_3/4/5.
  - [ ] **Not yet covered: a swap where both tracks stay alive.** Everything above
        acts on reconciliation, which only runs for unmapped track_ids. If BoT-SORT
        keeps both tracks through the crossing and simply exchanges the bodies, no
        reconciliation happens and nothing here fires. Detecting that needs a
        per-track motion-consistency check (a discontinuity in one track mirrored
        by the inverse in another), and measuring it needs hand-labeled crossing
        events — the synthetic tests can't tell us the real rate.
  - [X] **Duplicate boxes on one player — fixed in Plan 2.7 below.**
  - [ ] **Not yet measured on real footage.** The counters above show the
        machinery firing, not that it fires *correctly*: `switch_log` counts
        reconciliations absorbed and is blind to swaps. A swap-specific benchmark
        (hand-labeled crossings on match_4.mp4, asserting post-crossing identity)
        is the prerequisite for tuning `MAX_NORM_DIST`/`SIZE_RATIO_RANGE` against
        anything better than intuition.

- [X] **Plan 2.7: One Body, Two Boxes**

  Found while inspecting the match_5 swap frame-by-frame: the trigger was not a
  crossing at all. YOLO returns both a box around a running player (wide, because
  an extended leg stretches it) and a second box around just their torso —
  verified by eye at 8x zoom on f1441-1447, one body, two rectangles. BoT-SORT
  gives the second one its own track_id, so one player reaches the pipeline as
  two: two markers, two team votes, and in Plan 3 two sets of distance/heatmap
  numbers for one person.

  - [X] **Geometric test, not an IoU threshold.** A split is one box horizontally
        inside the other (intersection >= 0.9 of the smaller) while both share a
        top AND a bottom edge within 5% of the smaller box's height. Two different
        people cannot share a head line and a feet line to within a few percent,
        so it does not fire on a player occluded behind another — measured on
        match_5, 114 contained pairs share both edges, ~106 share only the feet
        line, and the latter are left alone. Plain NMS cannot separate these: the
        split pairs' IoU runs 0.34-0.84 (median 0.62), straight through the range
        where real overlapping players live.
  - [X] **Age decides which track survives, not size.** The established track was
        the *smaller* box in 42 of those 114 pairs, so keeping the bigger box
        would pick wrong more than a third of the time. Dropping the newer of the
        pair suppresses the short-lived phantoms (track 57 for 30 of its 94
        cycles, track 67 for 16 of 24) and barely touches real tracks (2 cycles
        out of 1595).
  - [X] **Alias the phantom track, never delete its box.** Deleting was measured
        to be far worse than the problem: the track vanishes, the player is marked
        lost, and next cycle it returns unmapped and gets reconciled — on match_5
        that took reconciliations 1 -> 28 and reintroduced 5 identity swaps. So
        `SplitDetectionSuppressor` rewrites the phantom's track_id to the track it
        belongs to instead. One stable track, no identity minted for the split,
        and the alias still resolves on the later cycles where the phantom is the
        only box on that player. An alias is revoked if the two tracks are ever
        seen apart, since that proves they were two real detections.
  - [X] Measured on match_5.mp4 (realtime, A/B in the same session): persistent
        player IDs 61 -> 51-53, tracks changing owner 0 in both, reconciliations
        unchanged at 2, 103 split detections collapsed across 25 tracks. No
        latency cost — worker total averaged 20.2ms with it and 20.4ms without,
        inside run-to-run noise. Frame-drop counts swung 24-90 across four runs
        regardless of the setting, so they say nothing about this change.
  - [X] Tests: `tests/test_split_detections.py` (9 synthetic cases, including the
        real f1443 geometry, the occluded-player-behind case that must survive,
        the phantom-seen-alone case that deleting got wrong, and alias revocation).
  - [ ] **Not covered: the wide box is still the surviving geometry sometimes.**
        When the leg-extended box is the established track, the player's bbox
        centre is pulled sideways by the leg. Harmless for identity, but Plan 3
        should take position from the box bottom-centre rather than its centroid.

- [ ] **Plan 3: Player & Match Analytics**

  - [ ] `PlayerState`: per-track position (pitch-relative, once calibration exists),
        velocity, team, distance covered, speed — built on top of the existing
        tracking + team classification pipeline
  - [ ] Heatmaps per player (and per team)
  - [ ] Possession detection (nearest player to ball, possession changes over time)
  - [ ] (Longer-term, not yet scoped in detail) formations/team shape, passing networks
