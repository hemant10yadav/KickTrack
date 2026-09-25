"""Regression tests for the two problems found debugging Plan 3 (see docs/PLAN.md
Plan 3.5): the async worker dropping frames it can't keep up with, and BoT-SORT
churning a player's track_id (fixed by PlayerIdentityManager). Runs the real
pipeline against fixture footage, so it's marked `slow` like test_pipeline_fps.py.
"""

from pathlib import Path

import pytest
from ultralytics import YOLO

from scripts.pipeline import MODEL_NAME, InferenceWorker, PlayerTracker

# match_5.mp4 (50fps, 20ms budget) was the demanding case under yolov8m (~24-25ms
# avg inference, structurally over budget -- some drops were unavoidable no matter
# how the rest of the pipeline was tuned). Switching to yolov8s at the same
# 640x1152 imgsz (see MODEL_NAME in scripts/pipeline.py) dropped inference to
# ~18ms avg, clearing the budget outright -- match_5 now drops 0 frames like the
# other clips, so it no longer needs its own looser ceiling.
MAX_DROP_RATIO = {
    "match_3": 0.05,
    "match_4": 0.05,
    "match_5": 0.05,
}

# Loose ceiling on persistent player_ids for a ~10-20s clip with ~22-28 real people
# in frame (players + refs), generous enough to cover legitimate sideline/staff
# detections without masking a real churn regression (pre-fix baselines were
# 2-3x higher: see docs/PLAN.md Plan 3.5 for the raw BoT-SORT track_id counts).
MAX_PLAYER_IDS = {
    "match_3": 70,
    "match_4": 45,
    "match_5": 65,
}

MAX_SWAP_CORRECTIONS = {
    "match_3": 5,
    "match_4": 60,
    "match_5": 25,
}

FIXTURE_VIDEOS = [
    Path("tests/videos/match_3.mp4"),
    Path("tests/videos/match_4.mp4"),
    Path("tests/videos/match_5.mp4"),
]


def _run(video_path: Path) -> tuple[InferenceWorker, PlayerTracker]:
    model = YOLO(MODEL_NAME)
    tracker = PlayerTracker(video_path, model, show_window=False)
    worker_holder = {}
    orig_run = tracker._play

    def _play_and_capture(cap, worker, calibration_worker, pacer):
        worker_holder["worker"] = worker
        return orig_run(cap, worker, calibration_worker, pacer)

    tracker._play = _play_and_capture
    tracker.run()
    return worker_holder["worker"], tracker


@pytest.mark.slow
@pytest.mark.parametrize("video_path", FIXTURE_VIDEOS, ids=lambda p: p.stem)
def test_frame_drop_rate_does_not_regress(video_path):
    if not video_path.exists():
        pytest.skip(f"fixture video not found: {video_path}")
    if not Path(MODEL_NAME).exists():
        pytest.skip(f"CoreML model not found: {MODEL_NAME} (export it first, see CLAUDE.md)")

    worker, _ = _run(video_path)
    stats = worker.stats
    drop_ratio = stats.frames_skipped / stats.frames_submitted

    max_ratio = MAX_DROP_RATIO[video_path.stem]
    assert drop_ratio <= max_ratio, (
        f"{video_path}: dropped {stats.frames_skipped}/{stats.frames_submitted} "
        f"frames ({drop_ratio:.0%}), above the {max_ratio:.0%} regression ceiling "
        f"(avg inference was {sum(stats.inference_ms) / len(stats.inference_ms):.1f}ms)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("video_path", FIXTURE_VIDEOS, ids=lambda p: p.stem)
def test_player_id_churn_does_not_regress(video_path):
    if not video_path.exists():
        pytest.skip(f"fixture video not found: {video_path}")
    if not Path(MODEL_NAME).exists():
        pytest.skip(f"CoreML model not found: {MODEL_NAME} (export it first, see CLAUDE.md)")

    _, tracker = _run(video_path)
    unique_player_ids = tracker.resolver.identity.total_players_minted

    max_ids = MAX_PLAYER_IDS[video_path.stem]
    assert unique_player_ids <= max_ids, (
        f"{video_path}: minted {unique_player_ids} persistent player_ids, above the "
        f"{max_ids} regression ceiling -- possible track_id churn regression "
        f"(BoT-SORT tuning in botsort_custom.yaml or PlayerIdentityManager gates)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("video_path", FIXTURE_VIDEOS, ids=lambda p: p.stem)
def test_body_swap_corrections_stay_bounded(video_path):
    """Every owner change of a BoT-SORT track is now a deliberate correction of
    a swap the tracker made silently (docs/PLAN.md Plan 2.8): the track's box
    stopped wearing its player's jersey and another body was found wearing it.
    The hand-checked swaps themselves are pinned in tests/test_identity_replay.py
    (drop-free, deterministic); this only guards against the correction
    machinery running away on real, drop-prone footage. Measured when Plan 2.8
    landed: 10 on match_5, 22 on match_4 (drop-free), 0 on match_3.
    """
    if not video_path.exists():
        pytest.skip(f"fixture video not found: {video_path}")
    if not Path(MODEL_NAME).exists():
        pytest.skip(f"CoreML model not found: {MODEL_NAME} (export it first, see CLAUDE.md)")

    _, tracker = _run(video_path)
    corrections = tracker.resolver.identity.swap_log

    assert len(corrections) <= MAX_SWAP_CORRECTIONS[video_path.stem], (
        f"{video_path}: {len(corrections)} live tracks were moved to another player, "
        f"above the {MAX_SWAP_CORRECTIONS[video_path.stem]} ceiling -- the jersey gates "
        f"in PlayerIdentityManager may have loosened: {corrections[:10]}"
    )
