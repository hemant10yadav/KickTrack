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

- [ ] **Plan 3: Player & Match Analytics**

  - [ ] `PlayerState`: per-track position (pitch-relative, once calibration exists),
        velocity, team, distance covered, speed — built on top of the existing
        tracking + team classification pipeline
  - [ ] Heatmaps per player (and per team)
  - [ ] Possession detection (nearest player to ball, possession changes over time)
  - [ ] (Longer-term, not yet scoped in detail) formations/team shape, passing networks
