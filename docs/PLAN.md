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
  - [X] **A swap where both tracks stay alive — fixed in Plan 2.8 below.** Everything
        above acts on reconciliation, which only runs for unmapped track_ids; the
        swaps actually on match_5 never produce one.
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

- [X] **Plan 2.8: Body Swaps on Live Tracks (appearance-anchored identity)**

  The user reported the goalkeeper's ID walking off with a player who came near
  him on match_5.mp4, and outfield IDs swapping when players came together.
  A drop-free per-frame dump of the pipeline (raw BoT-SORT boxes + jersey color
  per box) found both, frame-exact, and neither is a crossing in the Plan 2.6
  sense: the detector returned **one box covering two bodies**, BoT-SORT kept its
  `track_id` on that box, and when the box shrank back it was on the *other*
  body. No track went lost, so nothing in Plan 2.6 could fire.

  - f586-595: one tall box spans the orange goalkeeper and a white defender
    standing in front of him; f596 the box (still track 10) shrinks onto the
    defender; f600 the goalkeeper gets a brand-new track 40 -> new player id.
  - f1607-1636: blue player 22 and white player 8 run together; f1629 track 22's
    box is on the white body; f1633 the blue body gets track 69 -> new id; f1637
    track 8 dies. Net: the blue player's whole history now belongs to the white
    one.
  - f172: a third pair (17 white, 19 blue) whose tracks BoT-SORT exchanged with
    *both* alive — the case Plan 2.6 listed as not covered.

  - [X] **Anchor every identity to a jersey color.** `PlayerState.appearance` is a
        slow EMA (0.9) of the torso color on trusted sightings. `JerseySampler`
        (replacing `TeamGate`) samples each box once per cycle; cost measured at
        ~0.4ms per cycle for all boxes, worker total unchanged (20.7ms avg on
        match_5).
  - [X] **Relative jersey test, not an absolute threshold.** A sighting is a
        *mismatch* only when it is >40 BGR units from its own player *and* some
        other known player's appearance is at least 2x nearer (`_wears_another_jersey`).
        match_5 same-body noise is p99 ~20-35 with kits ~75 apart (blue/white)
        and ~190 (goalkeeper/white), but match_4 is far noisier (p95 40-90:
        shadows, crowd behind the crop, and a yellow kit that the grass mask
        half-eats). An absolute threshold there minted 212 ids instead of 155 and
        made 106 spurious "corrections"; the relative test brought it to 150/22.
        A color that is itself pitch green (`is_grass_color`) is no jersey at all
        and is dropped by `extract_jersey_color`.
  - [X] **Suspect tracks re-enter the joint assignment.** After 3 consecutive
        judged mismatches a live track becomes a row next to the unmapped ones,
        with its own player and the recently-lost players as candidates, so a
        both-alive exchange resolves as one Hungarian solution. Cost is
        normalized position + a capped color term; a transfer away from an
        established owner needs cost <= 2.0 (unmapped: 3.0).
  - [X] **Young identities yield.** The abandoned body usually gets a fresh
        track a few cycles *before* the stolen one is caught (t69 at f1633 vs.
        t22 suspect at f1640), so an identity minted within the last 20 cycles is
        re-examined whenever a suspect or a just-lost player exists, and is
        retired if an established player fits its box.
  - [X] **An established owner only yields when its body is accounted for.**
        A box that swallowed a second body can read as the other jersey for as
        long as they overlap (match_4: white hidden behind yellow), and nothing
        says which body the track follows when they part — acting then produced
        oscillating transfers and one permanent swap. A suspect transfer is
        therefore only accepted if the displaced owner is claimed by another row
        of the *same* assignment (`_drop_unaccounted_transfers`): the stolen
        track's old body, now detected on its own. That makes the fix
        evidence-driven — never earlier than the moment the second body shows up.
  - [X] **Color is only judged on single-body boxes.** Not while overlapping
        another live detection, not when >1.3x taller than the player, and not
        when wide/tall enough (1.4x) to hold the owner plus a swallowed lost
        player. A merged box's color is a blend and says nothing about ownership.
  - [X] **A swallowed player stays pending and rides inside the box.** Before,
        the "swallowed" flag only lasted the cycle a track vanished, so a merged
        box was treated as clean from the second frame on, and the swallowed
        player expired after 30 cycles even though the box holding them was still
        there (the 17/19 merge lasted ~70 frames and moved 80px). Now a lost
        player covered by a live box keeps `last_seen_frame` fresh and its
        position follows that box, so it is matched where the merge ends.
        Merged boxes are flagged occluded for their whole duration
        (`occluded_player_ids`), which is what Plan 3 analytics should key off.
  - [X] **Identity resolution moved off the inference thread.** The first
        version ran all of this inside `InferenceWorker`, and the ~0.7ms it
        added per cycle (jersey sampling ~0.25ms, occlusion bookkeeping, the
        assignment) was enough to matter: on the 10s match_5 fixture inference
        alone averages 19.9ms of the 20ms budget, and worker drops went 24/502
        (4.8%, already at the 5% ceiling) -> 38/502 (7.7%). Two things fixed
        it. The pairwise-overlap bookkeeping was vectorized (`pairwise_overlap`,
        one numpy op instead of ~700 Python `boxes_overlap` calls; that loop was
        72% of the identity layer's time) and the torso median uses
        `np.partition` (half the cost of `np.median` on tiny crops). Then the
        whole step — split suppression, identity, jersey sampling, confirm/grace
        state — became `IdentityResolver`, run on the *display* thread once per
        new worker result: that thread idles ~13ms per frame, and the worker
        now does only inference + box extraction. Full match_5 realtime: worker
        drops 90 -> **0**, worker total 21.1 -> 18.8ms avg; resolution costs the
        display loop ~2ms per new result (3 of 1750 frames over the 20ms display budget). The 10s
        fixture: 13-20/502 drops, under the baseline's 24.
  - [X] **Measured.** Drop-free replay of match_5: goalkeeper keeps id 10 at
        f300/650/1000; blue/white keep 22/8 at f1600 and f1660; 17/19 come apart
        as 17/19 at f1748. 52 ids minted (was 53), 13 corrections. Realtime CLI
        run (paced, 0 frames dropped): same probes all correct, 52 ids, 14
        corrections. match_4 (drop-free): 150 ids (was 155), 22 corrections —
        spot-checked strips (f132, f410, f1571) are right or restore an earlier
        silent swap. match_3: unchanged, 0 corrections. Full slow suite (fps,
        drop rate, id ceilings, correction ceilings on all three clips) passes.
  - [X] Tests: `tests/test_identity_replay.py` replays a recorded fixture
        (`tests/fixtures/match_5_tracks.jsonl.gz`, 380KB: raw boxes + colors for
        all 1750 frames) through the real split/identity/team code with no YOLO
        or video, and pins the three hand-checked pairs above — fast and
        deterministic, unlike the realtime slow tests. `tests/test_player_identity.py`
        gained jersey-gate cases (another player's jersey refused, same jersey
        merged, odd lighting alone still merged). `test_no_tracker_id_changes_owner`
        is gone: a track changing owner is now by construction a logged
        correction, replaced by a bounded-corrections ceiling per clip.
  - [ ] **Known limit: unclaimed deviations are adopted after 30 cycles.** A
        track whose color disagrees with its player for 30 judged cycles with no
        candidate to claim the old body adopts the new color as a lighting
        change (`ACCEPT_DEVIATION_AFTER`). On match_4 this legitimized at least
        one swap the machinery never got evidence for (a yellow player's id ending
        on a white body around f1571). Raising it risks freezing on real lighting
        changes; the right fix is an appearance descriptor less sensitive to
        shadow (chromaticity rather than BGR), measured on match_4.
  - [ ] **Known limit: a merged box carries one id.** While two bodies share
        one detection the box is shown under whichever id owned it going in;
        that is flagged occluded, not resolved — there is nothing to resolve it
        with until the detector separates them.

- [X] **Plan 2.9: A Detector That Keeps Two Boxes on Two Bodies**

  Plan 2.8 corrects body swaps after the fact; every one of them began with the
  detector returning a single box over two overlapping players. This plan asked
  whether that can be fixed upstream, and measured instead of guessing.

  - [X] **The detector was not the whole story.** Raw v8s detections on the three
        match_5 merge episodes (no tracker): keeper+defender 16/19 frames with a
        box per body, blue22+white8 34/34, white17+blue19 24/73. The *tracked*
        output had 3/19, 29/34 and 1/73. The gap is BoT-SORT's `new_track_thresh`
        (0.5, raised in Plan 2.5 against churn): the second body's box sits at
        0.17-0.47 confidence, so no track ever starts for it, and the existing
        track stays on the tall box that covers both.
  - [X] **Lowering the threshold is not the answer.** At 0.35 the merges separate
        earlier (keeper 11/19, hardest pair 23/73) but raw BoT-SORT ids go 63 ->
        144, minted ids 52 -> 81, and the blue/white probe breaks (the white
        player keeps getting fresh ids). At 0.25: 177 raw ids, 93 minted, two
        probes broken. Same conclusion as Plan 2.5, now on the identity layer's
        stronger footing.
  - [X] **yolov8m separates no better** (keeper 10/19 raw) at 18.3 vs 11.9ms
        predict-only. Size is not the lever.
  - [X] **YOLO26s is.** NMS-free head, exported at the same 640x1152. Raw:
        19/19, 34/34, 24/73. Tracked, with `botsort_custom.yaml` unchanged:
        19/19, 34/34 (every frame two boxes, no third), 24/73. Full match_5:
        raw BoT-SORT ids 76 (v8s 63), minted 46-48 (v8s 50-53), reconciliations
        3 (14), all ten identity probes correct. YOLO11s separates similarly
        (19/19, 34/34, 20/73) but mints 58 ids and breaks the blue/white probe.
  - [X] **Recall verified visually, not by count.** Raw detections of v8s and
        26s drawn on match_5 f600/f1200 and match_4 f300/f1500: every on-pitch
        player boxed by v8s is boxed by 26s; the count differences (28 vs 29,
        26 vs 28, 28 vs 26, 31 vs 32) are v8s's low-confidence duplicate boxes on
        bodies it already had, and sideline people.
  - [X] **Latency: +0.5-0.7ms per cycle.** Controlled, same 200 fixture frames,
        back to back: predict-only 13.0 vs 12.5ms, through `track()` 19.2 vs
        18.5ms (p95 21.4 vs 20.5). Realtime on the full clip, alternating runs
        with the machine at load average ~4: 20.2 vs 19.8ms, 26 vs 23 of 1750
        frames dropped. On a quiet machine v8s ran at 18.8ms with 0 drops
        (Plan 2.8), so 26s should sit ~19.4ms: under the 20ms budget with less
        headroom. `test_frame_drop_rate_does_not_regress[match_5]` could not be
        validated at the time (Chrome pushed the load average to 6-8 and v8s
        itself failed it at 21.7-22.7ms) -- re-run it on a quiet machine. Side
        finding: BoT-SORT + sparse-optical-flow GMC costs ~6ms of every cycle on
        top of inference, for either model; that is the next latency lever.
  - [X] **Container boxes.** 26s returns, for two overlapping players, a box
        each *and* a box around both, on 35/1439 realtime frames (v8s: 6). Three
        detections for two people; the union carried a third marker (and, at
        f588-592, the keeper's own established track). `SplitDetectionSuppressor`
        now drops a box holding two clearly shorter boxes; the track it carried
        is reconciled onto the right body by the identity layer from that body's
        own box. A box holding one shorter box (a player behind another) is
        untouched. Realtime match_5 with 26s + this rule: 32 containers dropped
        (4 borderline ones remain), 42 ids minted (v8s 50-53), 7
        reconciliations, all ten probes correct. Three synthetic tests in
        `tests/test_split_detections.py`.
  - [X] **Export gotcha.** coremltools 9.0 cannot convert YOLO11/26 under the
        project's numpy 2.5 (`int()` on a 1-element array, then a const of Vars);
        ultralytics itself warns it wants numpy<=2.3.5. Exporting inside
        `uv run --with "numpy==2.3.5"` works and leaves the project env and
        lockfile untouched. The v8 exports never hit those ops.
  - [ ] **Still merged: two bodies directly in line.** The white17+blue19 pair
        stays one box for ~46 frames with every model tried (one player almost
        fully behind the other). That is a detector limit at this resolution; the
        identity layer's swallowed-player handling (Plan 2.8) covers it after
        they part.
  - [ ] **Not tried: a football-specific fine-tune** (e.g. the Roboflow
        football-players-detection set). Worth it only if in-line merges turn
        out to matter for Plan 3 numbers; those frames are already flagged
        occluded and excluded from position samples.

- [X] **Plan 2.10: Smooth 4K Playback, Synced Markers, Live Streaming**

  Goal: play and stream 4K sources at their native frame rate with no drops, and
  markers on the players in the frame shown rather than a cycle behind. Measured
  on a 4K (3840x2160) upscale of match_5 (50fps, 20ms budget); the repo's clips
  are 1080p, so 1080p numbers are the same match_5.
  - [X] **Measured first: the window, not the AI, broke 4K.** Per-frame costs at
        4K: decode 2.6ms, inference ~14.5ms (the model sees 1152px either way),
        but `cv2.imshow` + `cv2.waitKey` ~30ms (1080p ~18ms) -- the repaint
        alone was over budget. Old code, full 4K clip: 20.4 display fps, 275/1750
        frames over budget, worker 36ms.
  - [X] **`FrameScaler`: shrink once after decode to `--display-width` (1920).**
        Everything -- detection, identity, drawing, window, output -- runs at
        that size; 1080p sources pass through untouched. INTER_LINEAR (0.4ms;
        identical to INTER_AREA on an exact 2:1, which costs 5.3ms). Detections
        on 150 frames, 4K straight into the model vs downscaled: 4055 matched
        boxes, IoU 0.979, no box of conf >= 0.5 lost (22 vs 3 unmatched boxes,
        all conf 0.25-0.5). Calibration keyframes keep the source frame (see
        the resize warning in `CalibrationWorker.submit`); their homography is
        rescaled (`homography_to_working`), checked against calibrating the
        1080p original of the same frame: 0.1-0.4px apart on frame 440.
  - [X] **Calibration hand-off stalled the display thread.** `submit` pulled
        the previous, unstarted frame back out of the multiprocessing queue to
        replace it ("latest wins"), unpickling a whole frame on the display
        thread -- on every frame of a 0.3-1.8s calibration: 26ms per 4K frame
        (~6ms at 1080p), 156 of the first 300 frames. Now a frame is only
        submitted to an idle, warmed-up child (it sends `CALIBRATOR_READY`),
        which starts on it at once -- as fresh, no displacement. 4K full clip,
        OpenCV window: 38.7 display fps, 6/1750 frames over budget, worker 20ms.
  - [X] **`FfplayViewer` (`--viewer ffplay`, default when installed).** Even a
        540p OpenCV window costs ~14ms per `waitKey` on macOS (1080p 17.7ms):
        a fixed repaint wait that capped the loop at ~40-45fps. ffplay draws
        on the GPU in its own process; the loop enqueues the frame and paces
        with a sleep. Full match_5: 49.1 fps (OpenCV window 44.9).
  - [X] **`PlaybackDelay` + `ResultTimeline` (`--display-delay-ms`, default 100).**
        Frames are held back 100ms (5 frames at 50fps) so AI results exist on
        both sides of each one, and its boxes are interpolated onto that exact
        frame in video time; newer-than-every-result frames extrapolate from
        the two newest (replaces the wall-clock `MotionExtrapolator`). Synced
        boxes are not eased (easing would re-add a frame of lag). Pitch
        overlay/ball ring use the homography propagated for the frame shown.
        Full clips: 1750/1750 frames drawn from their own detections, 1080p
        and 4K.
  - [X] **`FfmpegOutput`: `--output` via ffmpeg's `h264_videotoolbox`, to a file or
        a stream URL (rtmp/srt/udp/rtsp; streams are paced).** cv2's `mp4v`
        took 10.6ms per 4K frame on the display thread. Checked: 600-frame
        file is h264 1080p50 with all 600 frames; a udp:// stream decoded 400
        frames on the receiving end in its 8s window.
  - [X] **GMC at downscale 4 (`GMC_DOWNSCALE`).** Once display ran at a true 50fps
        the worker (20.8ms) fell just behind (82/1750 skipped). BoT-SORT's
        sparse-flow GMC cost 4.5ms at ultralytics' hardcoded downscale 2, 2.5ms
        at 4, shift estimate within 0.13px mean / 0.7px max. Worker 18.6ms,
        skipped 18/1750 (1080p) and 34/1750 (4K); 41 ids minted and 9 swap
        corrections vs 43 and 12 on the old code's realtime run. All 14 slow
        regression tests pass (churn, drop-rate ceilings, fps). Not yet
        re-checked with frame strips around each logged correction.
  - [X] **Int8 weights: not adopted.** `quantize=8` for CoreML is k-means weight
        palettization only (activations stay fp16), exported with
        `uv run --with scikit-learn --with numpy==2.3.5` (no project dependency
        change). 12.0 vs 12.3-12.7ms predict (~0.5ms), boxes IoU 0.977, no
        box of conf >= 0.5 unmatched -- too small a gain to re-verify the
        identity probes and the ball's low-confidence detections for.
  - [X] **Seen here, fixed in Plan 2.11: propagated pitch overlay drifts between
        keyframes** (a halfway line drawn straight where the true one slants
        ~50px on frame 520 of the 4K run). Both the 1080p and the rescaled 4K
        calibration of that frame get it right, so it is propagation, not the
        rescale.

- [X] **Plan 2.11: Pitch Overlay Drift Between Keyframes**

  Measured against a full PnLCalib calibration of every 10th frame (match_5:
  175 of 176; match_4: 376 of 376), frames whose calibration jumps >40px from
  a neighbour's excluded as reference glitches. Replay (`HomographyWorker`
  driven synchronously): propagate every frame, a keyframe every 90 frames
  arriving 25 frames late, raw calibrations as keyframes (glitches included),
  over 9 keyframe phases (match_5) / 3 (match_4). Calibrations 10 frames
  apart differ ~9.5px median, which is the floor this can resolve.
  - [X] **Cause 1: frame-to-frame chaining at a 3px RANSAC threshold.** A pan
        moves the picture 1-3px per frame, inside the threshold, so points that
        don't move with the pitch (scoreboard, players) counted as inliers and
        biased every step toward no motion. Over 90 frames: 20px median, 73px
        p90, 106px max. Fix: measure each frame's motion from an anchor up to
        10 frames back, 1px threshold, re-anchor every 10 frames: 6.7 / 14 /
        26px. (Tried and worse: corners on grass only -- 45px median; grass
        has too little texture.)
  - [X] **Cause 2: corners came from the broadcast graphics.** The strongest
        200 corners of the frame were, on match_4, nearly all on the ticker,
        scoreboard and logo: frames 1400->1410 the calibrations have the
        camera moving 30px, propagation 3px, with 103 of 111 points agreeing on
        it. Fix: 5 corners per cell of an 8x5 grid (found at half size: 10.7 ->
        3.2ms per anchor, same drift). match_4 >50px frames 203 -> 4 of 849.
  - [X] **Cause 3: a blurred fast pan left too few 1px inliers**, and propagation
        froze until the next keyframe (match_4 1449-1555, up to 828px). Fix:
        retry at 3px. Re-measuring from the last good frame instead was tried:
        160px max alone, no gain on top of the retry.
  - [X] **Cause 4 (Plan 3.1's "double-snap"): keyframes restarted propagation
        from their 0.3-1.8s-old frame.** `HomographyWorker` now keeps the
        homographies it propagated and applies a keyframe as a pitch-side
        correction on top of what it had for that frame (`rebase`); only a
        keyframe with no propagation near its frame restarts from it.
  - [X] **Cause 5: PnLCalib glitches were accepted as keyframes** (frames
        1110/1130 of match_5 land 205-287px off). A keyframe more than 3% of
        the frame width from propagation is held back; the next one confirms it
        if it lands in the same place (propagation had drifted) -- two
        disagreeing ones are both held.
  - [X] **Result (replay, old -> new).** match_5: median 6.6 -> 5.7px, p90 18.5
        -> 14.5, max 256 -> 36, frames >50px 40 -> 0 of 1512 (2 glitch
        keyframes held back of 172). match_4: median 28.0 -> 8.8px, p90 247 ->
        22, max 1290 -> 49, >50px 328 -> 0 of 849. Cost: 1.9 -> ~2ms per
        propagated frame avg on the homography thread (anchors 3.2ms, every
        10th frame). Realtime match_5: 49.2 fps either way; analytics
        "calibration shifts cancelled" 75 -> 39. Frame strips 520/1130/1440/
        1600: halfway line and boxes on the painted lines.
  - [ ] **Left:** isolated reference frames on match_4 (770, 820, 1030) sit
        400px+ from the overlay; they look like calibration glitches in the
        reference, not checked by eye. Distances/top speeds re-measured: see
        Plan 3.1's last item (no top speed over 10 m/s left on either clip).

- [ ] **Plan 3: Player & Match Analytics**

  - [X] Per-player pitch-relative position, distance covered, speed (Plan 3.1)
  - [X] Heatmaps per player (and per team) (Plan 3.1)
  - [X] Possession detection and team pass counts (Plan 3.2)
  - [X] Pass counter that holds up on real footage (Plan 3.3)
  - [X] Team shape: width, depth, area, defensive line, in and out of possession (Plan 3.4)
  - [X] Formation lines: defence, midfield and attack of the team without the ball (Plan 3.5)
  - [ ] (Longer-term, not yet scoped in detail) passing networks

- [X] **Plan 3.5: Formation Lines**

  The team without the ball gets its defence, midfield and attack drawn as
  dots at its players' feet joined across the pitch, one colour per line,
  and its line counts ("4-4-2") in the panel; the team with the ball keeps
  its outline. `FormationLines` (`scripts/analytics.py`), fed by
  `TeamShapeAnalytics`; drawn by `TeamShapeOverlay`. Prototyped offline on
  the five recordings before any drawing, and checked on frames and on
  half-second contact sheets of fresh realtime runs.

  - [X] **Lines from where players are now, not a named formation.** A fit
        of fixed formations (4-4-2, 4-3-3, 5-4-1, ...) to 15 s averages was
        tried first: after fixing a bug (it kept the 10 most *advanced* ids
        when stale ones were in the window) and adding hysteresis it was
        stable (2-3 changes a minute), but wrong on the video -- match_5's
        City pressed with two centre backs on halfway and three players
        6-10 m ahead, which it could only call a back five. Now each
        player's depth in front of his team's last man (so the team moving
        as one moves no one between lines) is smoothed over 1 s, and the
        sorted team is cut into three where the cut leaves least spread
        inside the lines. match_5 City: 4-2-4, 2-3-5 and 3-2-5 while pressing
        (12-24 s), 5-4-1 / 6-3-1 / 4-4-1 once camped in their box (24-35 s);
        Tottenham 4-3-3 when they come into view.
  - [X] **Stable without lagging.** A new cut replaces the lines only when
        it fits 20% better for 1 s: a full back hovering across the cut
        point switches lines 5 times in 6 s without it, 0 with it. Lines
        change ~10 times a minute on the long clips -- the players moving,
        not flicker, on the contact sheets.
  - [X] **Possession held while the ball travels.** The ball is at a
        player's feet in 15-30% of results; a team now keeps possession
        until the other one has it for 1 s (or 10 s with nobody on it). The
        1 s stops a misread touch moving the lines to the other team and
        back: team swaps per minute match_5 3.6 -> 0, match_6 4.1 -> 1.6,
        match_8 6.0 -> 1.3. Lines are drawn 54-68% of the time on each clip.
  - [X] **Cost:** 2.0 ms average draw and 40 of 1750 frames skipped on
        match_5, as before the lines.
  - [X] Tests: `tests/test_analytics.py` (+6: cut at the gaps, a player
        hovering between lines, a player who moves up for good, too few
        players and the gone, the "4-4-2" label, possession held and
        switched), `tests/test_shape_replay.py` (+2: City pressing with few
        back then a block of four or more; Tottenham's back four),
        `tests/test_pitch_overlay.py` (joined dots in line colours, outline
        while possession is unknown). The hysteresis and the possession
        switch were each checked to fail their test when switched off.
  - [ ] **Known limits.** Always three lines, so 4-2-3-1 reads as 4-5-1.
        A player's line needs his team's last man in view. A new id starts
        without a depth history and is put in the nearest line. Possession
        comes from the ball track, so a long spell with the ball unseen ends
        in outlines only.

- [ ] **Plan 3.4: Team Shape**

  Each team's width, depth, area and defensive line (its last outfield man's
  distance from its own goal), measured from the gated pitch positions every
  0.1 s, split by who has the ball, drawn on the video and written to
  `shape.json`. All in `TeamShapeAnalytics` (`scripts/analytics.py`);
  drawn by `TeamShapeOverlay` and `TeamShapePanel` (`scripts/display.py`).
  Recorded on all five clips (positions, teams, homography, possession per
  result) and checked on frames with each team's outline and line drawn on,
  at the moments the numbers moved.

  - [X] **Only what the camera can see.** The camera follows play, so a team
        is often cut off: on match_5 both teams' rearmost player sat on the
        halfway line for 15 s because that was the edge of the picture. A
        dimension is only measured when a point 5 m past the team on both ends
        of that axis (three points along each edge) projects into the frame;
        past a goal line or touchline only the line itself has to be seen.
        Otherwise it is "-", never a smaller number. Measurable share per
        team: match_5 91% / 22% depth, match_4 61 / 64%, match_6 84 / 96%,
        match_8 80 / 94%.
  - [X] **Which goal each team defends**, learned: defenders stand
        goal-side, so the team defending +x has its centroid further +x
        whether it defends or attacks (match_5: 6-10 m in every 3 s window).
        Committed after 2 s of both teams in view; right on every clip,
        checked against the keepers.
  - [X] **Teams from `TeamHistory`, not the latest read.** A Tottenham
        defender read as "other" for 3 s on match_4 dropped out of the shape
        and moved the line 20 m; `BallAnalytics.team_now` (the 10 s majority
        the pass counter already uses) keeps him.
  - [X] **Officials dropped.** Linesmen run the touchlines in colours near a
        kit: match_5's read sky blue and stretched City to 68 m wide; match_6
        had one on each team at once, match_4 one in Watford's yellow. A point
        within 1 m of a pitch line with no teammate within 30 m is dropped:
        the officials caught stood 33-44 m from "their" team, the nearest real
        player on a line (a Tottenham throw-in taker) 26-30 m. Then at most 10
        per team. Some seconds of official survive at 30 m; a lower threshold
        cut real wingers.
  - [X] **Numbers, matched to the video.** match_5 City: last man ~52 m from
        goal while pressing (11-24 s, on halfway), ~11 m camped in their box
        (31-34 s); 28 m wide without the ball vs 36 m with it. match_4
        Tottenham: 58 m wide in possession (a player on each touchline) vs
        40 m out. match_8 Tottenham: a 31 x 13 m low block at 30 s.
  - [X] **On the video**: a thin outline through the feet of the shape's
        players in the boxes shown on that frame (so it sits on them like
        the markers), and the defensive line touchline to touchline on the
        grass at the last man -- also found on that frame, from the same feet
        through the frame's own homography, as the line drawn from the 0.1 s
        samples (smoothed for the panel) trailed a defence stepping up. The
        second team is dashed, as two pale kits read alike.
        A "Team shape" panel (width / depth / line) sits on the minimap.
        `--hide-shape` turns both off. A faint tint inside the outline was
        dropped: blending its box of a 1920-wide frame cost 0.65 ms per team.
  - [X] **Cost on the display thread**, measured in isolation: recording
        0.19 ms per sample (10 per second -- every result, with a team vote
        per player, was 1.18 ms), outline + line 0.04 ms, panel 0.04 ms per
        frame. Drawn dashes are one `cv2.polylines` call (a `cv2.line` per
        dash took the draw from 2.5 to 6 ms).
  - [ ] **Not yet: a clean frame-drop comparison.** Realtime runs of match_5
        were too noisy to show an effect this small: baseline `main` alone
        skipped 28-193 of 1750 frames across seven runs as the machine heated
        up (and a VS Code language server took a core during some). The last
        run before the tint was dropped skipped 34.
  - [X] Tests: `tests/test_analytics.py` (+11: visible team, cut-off edge,
        edge on a pitch line, too few players, official vs winger on the
        touchline, 10-player cap, direction and last man, line waits for the
        direction, possession split, display hold, shape.json);
        `tests/test_shape_replay.py` (match_5 recorded with homographies,
        `tests/fixtures/match_5_shape.jsonl.gz`: direction, press then deep
        block, the linesman, narrower without the ball, Tottenham's depth
        unmeasured while cut off); overlay and panel in
        `tests/test_pitch_overlay.py`. Each synthetic guard was checked to
        fail with its rule switched off.
  - [ ] **Known limits.** Formations are not read yet. A team split between
        a kit and "other" by the classifier (match_7's striped Barcelona
        kit) gives a shape of the part that was read -- 7 m deep -- which the
        visibility check cannot catch; match_7 is mostly "-". The defensive
        line is one player, so a defender dropping onto the goal line moves
        it; it is the offside line only when the keeper is not the last man.

- [X] **Plan 3.3: A Pass Counter That Holds Up on Real Footage**

  Checked by eye on both clips: ball-following zoomed crops of the annotated
  realtime output (every 0.5 s, all of match_4, match_5's labelled moves),
  and the hand labels of match_5 (`tests/test_pass_replay.py`). The Plan 3.2
  counter booked keeper passes as lost, counted the referee, credited passes
  to the wrong team, and invented passes while one player dribbled alone.
  All fixes are in `PassCounter` / `TeamHistory` / `BallAnalytics`
  (`scripts/ball.py`); the tracker and possession rules are unchanged.

  - [X] **The panel.** Top-right, from the first frame to the last: "Passes
        completed" and a row per team (kit swatch, "white team", count), 0
        until a pass settles; before the teams are fitted they read "team
        1" / "team 2". Counts only ever go up.
  - [X] **Passes settle 2 s after they happen** (`SETTLE_S`) and are counted
        with the teams read around them, then frozen. Measured: a player's
        team read is wrong for seconds while he stands against someone else
        (match_5's 6, sky blue for 3 s beside a dark-coated steward while
        receiving), an id re-used for a new body carries the old kit until
        the colour average catches up (match_4's 65: yellow to 93.8 s, a white
        player from 98.3 s), and an id that swaps bodies in a tackle reads as
        the other kit from then on (match_4's 1, white to 49 s, yellow
        after). `TeamHistory` takes the majority from 10 s before to 2 s after
        the pass, within the player's current stint on screen (an absence
        over 1 s may be a new body), skipping the first 2.5 s back. The
        passer is judged at his own possession, not the receiver's.
  - [X] **Keepers.** They share `OTHER_TEAM` with the officials. A keeper
        within 25 m of a goal line is put on the side whose visible outfield
        players are on average deeper (defenders are goal-side): right at all
        8 keeper moments checked on both clips, where the nearest-centroid
        and deepest-player tests each failed some. Anyone else in
        `OTHER_TEAM` (the referee) is left out of the pass chain.
  - [X] **Set pieces.** A goal kick's keeper steps back for his run-up and
        never holds the ball; the player within 4 m of where a ball track
        begins (or is re-sighted after 0.3 s of coasting) is the passer when
        there is no recent possession to chain from.
  - [X] **Track jumps are not passes.** The ball track can hop onto a
        look-alike 20 m+ away. Real passes implied at most 27 m/s from one
        holder's feet to the next; jumps 35-48 m/s. So a pass over
        1 m + 30 m/s x the gap is skipped, and so is one whose receiver had
        the ball under 0.3 s after it arrived faster than 20 m/s (chained
        jumps; real first-time touches arrived at 7-15 m/s).
  - [X] **An opponent's brief touch between teammates** (under 0.3 s, the
        teammate has it within 1.5 s) is a pass that still reached a teammate
        -- match_5's pass through a City player's legs.
  - [X] **Measured.** match_5 labels (9 completed Tottenham passes, City
        losing the ball once): the recorded fixture books 8, the four fresh
        realtime runs 6-7, with no false or wrong-team pass in any of them
        (before: 5-6, with 2-4 false or wrong-team). The misses: the one-touch
        15 -> 8 -> 10 counts once, and the back pass at 3.5 s whenever
        calibration had not started yet. match_4 (no labels): every booked
        pass from 7 s to 150 s checked on the crops -- white 11 completed, 3
        lost; yellow 12 completed, 1 lost -- including 6 -> keeper, the
        keeper's long ball, yellow's back pass to its own keeper at 108 s,
        and the tackle scramble at 48-50 s.
  - [X] Tests: `tests/test_ball.py` (+7: jump, referee, settling with teams
        around the pass, team history across an id re-use and a swap, keeper
        back pass, goal kick); `tests/test_pass_replay.py` (match_5 labels on
        the recorded fixture, `tests/fixtures/match_5_ball.jsonl.gz`).
  - [ ] **Known limits.** A one-touch pass inside a quick exchange is
        usually merged into the next (possession needs 3 results), and a pass
        is only as good as the ball track: a pass played while the ball is
        unseen at both ends is missed.

- [X] **Plan 3.2: Ball, Possession and Team Pass Counts**

  `scripts/ball.py` (`BallTracker`, `PossessionTracker`, `PassCounter`,
  `BallAnalytics`), fed by `PlayerTracker` with each result's ball candidates
  projected onto the pitch and the players' gated positions. Live: the ball as
  a yellow ring on the video (hollow while coasting) and a dot on the minimap,
  a ring around the player in possession, and a top-right panel with each
  team's completed pass count. Teams are named by their fitted jersey colour
  ("white", "yellow", "sky blue" -- `color_name` / `TeamClassifier.team_name`),
  never by a cluster number; the short/long/lost breakdown is in the terminal
  summary and in `passes.json` (`--analytics-dir`). Team totals first;
  per-player counts next.

  - [X] **The ball comes from the pass we already run.** `model.track()` only
        returns tracked boxes and BoT-SORT never starts a track for the ball
        (a 10-17px blob at median confidence 0.12-0.20), so
        `BallCandidateCapture` is a predictor callback registered before the
        tracker's, which runs first and sees the raw multi-class detections.
        The predictor now runs with `classes=[0, 32]` and `conf=0.05`; person
        tracking is unchanged (BoT-SORT keeps its own thresholds) and so is
        latency (17.9 vs 18.1ms, same 200 fixture frames). Measured on both
        clips: an on-pitch candidate in 44% (match_5) / 55% (match_4) of
        results, consecutive ones within 3m 91-98% of the time, gaps median
        0.12s and 90% under 0.6s.
  - [X] **Tracker: follow sightings, coast with damped velocity, gate by ball
        speed.** A Kalman filter was tried first and dropped -- with a gate on
        top, the stale velocity after a ball stops at a foot kept dragging the
        estimate away from the sightings that said it had stopped, and no
        noise setting changed that. Now: position follows the accepted
        sighting, velocity is the displacement over the last 0.12s, coasting
        predicts with that velocity damped to 30%/s and gives up after 1.5s.
        A candidate is only admitted inside 2m + 35 m/s x time-since-sighting.
        Acquisition needs two sightings within 3m, preferring one at a
        player's feet.
  - [X] **Fixed features are not the ball.** The densest candidate cells on
        match_4 were on the touchline and at fixed spots, and one such blob
        held the track for seconds until a player walked past and "received"
        a pass. A track that moves under 1.5m in 2s with no player within
        2.5m is dropped and its spot banned for 20s (25-43 candidates so
        rejected per clip). A ball sitting at a player's feet is not affected.
  - [X] **Possession and passes.** Holder: nearest player within 1.5m of a
        ball seen within the last 0.3s, for 3 results in a row; released after
        3 results beyond 2.5m or when someone else takes it. A pass is
        possession moving to a different player within 4s: completed for a
        teammate, lost to an opponent; long at >= 30m (Opta's long-ball line).
  - [X] **Measured (realtime runs).** match_4: ball tracked in 89% of results,
        44 possessions, white (team 0): 5 completed / 9 lost, yellow (team 1):
        6 / 5, officials 3 "lost" (a keeper collecting the ball). match_5: 72%
        tracked, 16 possessions, sky blue (team 0): 0 / 2, white (team 1): 6
        completed (5 short, 1 long) / 3 lost. Clusters fitted as expected on
        both clips (kits as teams 0/1; orange keeper and dark officials as
        other). Strips around
        six match_4 events checked by eye: 22->6 (14s), 22->4 (31s), 6->8
        (47s), 8->1 (48s) are real passes; one "lost" at 17s was the touchline
        blob above (now suppressed; a static blob acquired less than 2s before
        a player arrives can still slip through).
  - [X] Tests: `tests/test_ball.py` (14: acquisition, coasting through a gap,
        loss after a long gap, a far look-alike, a fixed feature vs a ball at a
        player's feet, a kicked ball, possession hold/release, completed/lost/
        long passes, gap too long, candidates-to-pass end to end);
        BallRenderer smoke tests in `tests/test_pitch_overlay.py`.
  - [ ] **Not yet: precision/recall against hand labels.** The counts above
        are what the machinery produces, checked on a handful of events, not
        against a labelled clip. That labelling (20-30 moments per clip) is
        the prerequisite for tuning any threshold here, and for per-player
        pass counts and shots, which are the next Plan 3 steps.
  - [ ] **Known limits.** No ball, no event: a pass whose ball is unseen at
        both ends is missed; a pass played while either player is out of
        frame is missed and not guessed. The ball is only a candidate when the
        detector fires on it, and a fine-tuned detector (Plan 2.9's open item)
        is the lever if recall proves too low.

- [X] **Plan 3.1: Heatmaps and Distance Covered**

  `scripts/analytics.py` (`MatchAnalytics`, `PlayerTrace`, `PitchProjector`),
  fed by `PlayerTracker` once per new worker result with the confirmed boxes,
  the identity layer's occluded set and the current homography; written out by
  `--analytics-dir DIR` as `stats.json` plus a heatmap PNG per player and per
  team. Everything is in *video* time (frame id / fps), from the confirmed boxes,
  never the extrapolated/smoothed ones drawn on screen.

  - [X] **Feet, gated.** A player's position is the box's bottom-centre through
        the inverse homography. A sample is dropped (and counted) when there is
        no homography yet, when the identity layer flags the box occluded
        (contaminated by another body), or when it lands more than 3m outside
        the lines. match_5: 8.2k / 2.4k / 1.6k of ~34k samples.
  - [X] **Distance from 0.5s bucket means, chained within 1s.** Per-frame steps
        cannot be summed: the box bottom jitters 1-2px and swings with the
        running stride, and the median per-result step of a player is 6.5cm even
        when the calibration is still (3 m/s of fake motion at 50fps). Positions
        are averaged per 0.5s bucket and distance is the path between bucket
        means; a bucket only chains to the previous one within 1s, so time out of
        frame adds nothing. Measured on a logged run of match_5: 0.2s and 0.5s
        windows agree within 6% once the chain gap is not shorter than the window
        (a 0.5s gap with a 0.5s window silently halved everything -- the first
        bug found), so what survives 0.5s is real motion. Steps slower than 0.5
        m/s count as standing (residual drift of a stationary player is a few
        cm per bucket); steps faster than 12 m/s are identity or calibration
        errors and are dropped. Top speed is over two consecutive buckets (~1s).
  - [X] **Common-mode shifts are cancelled, not summed.** When the median move
        of all players between two results is >= 0.2m (no team averages 10 m/s
        in one direction), the mapping moved, not the players. The median shift
        is subtracted from the one step that straddles it; heat positions are
        left as measured. Measured on match_5: 91 such results of 1090. Breaking
        every player's motion chain there instead (the first attempt) discarded
        most of the clip: 45m instead of 90m for the top runner.
  - [X] **Finding: the keyframe hand-over double-snaps.** The big common shifts
        come in pairs 25 frames apart at every keyframe (~115 frames): a fresh
        keyframe homography arrives ~0.5s after its frame (calibration latency),
        `HomographyWorker.reset` restarts propagation from that *stale* frame,
        so everything snaps to a 0.5s-old camera pose and then catches up on
        the next propagate. Snap sizes 0.3-1m normally, 4m on a fast pan (frame
        1238). Cancelled here; the real fix is upstream -- compose the new
        keyframe homography with the propagation accumulated since its frame
        instead of resetting to it -- and is the next calibration task.
  - [X] **Sanity of the numbers.** match_5 (35s attacking phase, players in
        frame ~28s): 70-91m per outfield player, 2.5-3.3 m/s average, top
        8-10 m/s. match_4 (150s, 25fps): 220-380m for players tracked ~140s,
        1.6-2.7 m/s average, top 7-10 m/s. Averages are consistent with active
        play; top speeds sit at the human ceiling and are still slightly
        inflated by pan lag below the 0.2m gate. Heatmaps checked by eye:
        match_5 player 4's heat is the 60m run from centre-right to the left
        corner visible in his raw trajectory; match_4's player 17 spreads across
        the middle third like the central midfielder he is.
  - [X] Tests: `tests/test_analytics.py`, 14 synthetic cases through a known
        affine homography (projection, straight run, standing jitter, out-of-frame
        gap, impossible speed, gating counts, one result sampled once, heatmap
        mass, common shift cancelled while a runner keeps his distance, output
        files).
  - [ ] **Known limits.** Distance is only measured while a player is in frame
        (`tracked_s` says how long that was) and while the calibration is
        stable. Team ids are the classifier's clusters (2 kits + one for
        referees/staff), not named teams. No ball, so no possession, passes or
        shots yet -- those are the next Plan 3 steps and need ball detection
        first.
  - [X] **Live view.** `PitchMinimap` (scripts/display.py) draws a 315x204 top-down
        pitch bottom-right with a dot per player at their latest gated pitch
        position, team-coloured and labelled with the player id; each marker on
        the video carries its running distance ("42m") above the ID label. Cost
        measured at 0.19ms (minimap) + 0.03ms (captions) per displayed frame.
        Exposed a Plan 1 threshold: `TeamClassifier` waited for 25 distinct
        tracks before fitting, which the phantom-happy yolov8 tracker reached in
        seconds and the YOLO26 + identity-layer pipeline (~40 ids per clip) did
        not reach until late -- match_5 was still all-grey 24s in. Now 16.
  - [X] **Finding, fixed in Plan 3.2: k=3 team clustering collapsed the two
        kits on match_5.** With the keeper (orange) and touchline stewards
        (orange bibs) in the sample set, the three clusters came out as {sky
        blue + white}, {orange}, {dark officials}: the two kits are the closest
        pair of colours (~75 BGR apart vs ~190 to orange), so k-means merged
        them and every outfield marker rendered grey. `TeamClassifier` now fits
        four clusters, merges any pair closer than 40 BGR (one kit under two
        lights), and calls the two most populated groups the teams; keepers,
        officials and staff are `OTHER_TEAM`. `tests/test_team_classifier.py`.
  - [X] **Re-measured after the keyframe hand-over fix (Plan 2.11).** Realtime,
        old (pre-2.11) vs new code back to back, same machine. Common shifts
        cancelled: match_5 66 -> 25, match_4 374 -> 154. Top speed, players
        tracked >= 20s (match_5) / >= 100s (match_4): match_5 median 7.0 ->
        6.3 m/s, max 10.3 -> 8.6, over 10 m/s 1 -> 0 of 21; match_4 median
        9.8 -> 8.4, max 11.5 -> 9.9, over 10 m/s 4 -> 0 of 11 -- the
        "slightly inflated by pan lag" above was the snaps. Distance: match_5
        median 61 -> 70m (tracked time 26.6 -> 28.3s median as fewer samples
        lack a homography, 2.40 -> 2.58 m/s over it), match_4 296 -> 310m at
        2.41 -> 2.35 m/s -- about the same per second tracked.
