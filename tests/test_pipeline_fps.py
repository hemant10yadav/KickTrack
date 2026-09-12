"""Regression test: fail if the tracking pipeline's achieved display FPS drops.

Runs the real pipeline (real CoreML model, real footage) headlessly and checks
throughput against each fixture's own native frame rate. This exercises actual
hardware (Apple Silicon Neural Engine) and the gitignored exported model, so it's
marked `slow` and skipped by default — run explicitly with `uv run pytest -m slow`.
"""

from pathlib import Path

import cv2
import pytest
from ultralytics import YOLO

from scripts.track_players import MODEL_NAME, PlayerTracker

# FramePacer paces playback at the input video's own native fps, so achieved FPS
# can never exceed it — the right regression floor is a fraction of *that video's*
# fps, not a fixed number (a fixed floor would be meaningless for e.g. 60fps
# footage). Measured achieved FPS tracks native fps closely (~97-99%) when the
# pipeline is healthy, so 0.8 leaves room for normal machine variance while still
# catching a real regression (e.g. an expensive per-frame draw call, or something
# blocking the main display thread, which drags achieved FPS well below native).
MIN_FPS_RATIO = 0.8
FIXTURE_VIDEOS = [
    Path("tests/videos/match_3.mp4"),  # trimmed to 10s
    Path("tests/videos/match_4.mp4"),  # trimmed to 20s
    Path("tests/videos/match_5.mp4"),  # trimmed to 10s, 50fps -- the demanding case
]


def _native_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        cap.release()


@pytest.mark.slow
@pytest.mark.parametrize("video_path", FIXTURE_VIDEOS, ids=lambda p: p.stem)
def test_pipeline_fps_does_not_regress(video_path):
    if not video_path.exists():
        pytest.skip(f"fixture video not found: {video_path}")
    if not Path(MODEL_NAME).exists():
        pytest.skip(f"CoreML model not found: {MODEL_NAME} (export it first, see CLAUDE.md)")

    native_fps = _native_fps(video_path)
    min_acceptable_fps = MIN_FPS_RATIO * native_fps

    model = YOLO(MODEL_NAME)
    tracker = PlayerTracker(video_path, model, show_window=False)
    tracker.run()

    assert tracker.achieved_fps >= min_acceptable_fps, (
        f"Pipeline achieved {tracker.achieved_fps:.1f} FPS on {video_path} "
        f"(native {native_fps:.1f} FPS), below the {MIN_FPS_RATIO:.0%} "
        f"regression floor of {min_acceptable_fps:.1f} FPS"
    )
