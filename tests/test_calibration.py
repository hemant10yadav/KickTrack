"""Tests for PitchCalibrator -- single-frame pixel-to-pitch homography.

Uses tests/videos/match_4.mp4 (broadcast-style footage that Phase 0 verified
calibrates reliably -- see docs/PITCH_CALIBRATION_SPEC.md). Slow (loads a
~270MB model, runs real inference), like tests/test_tracking_quality.py.
"""

from pathlib import Path

import cv2
import pytest

from scripts.calibration import CalibrationResult, PitchCalibrator

FIXTURE_VIDEO = Path("tests/videos/match_4.mp4")
WEIGHTS_DIR = Path("weights/pnlcalib")


def _read_middle_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read a frame from {video_path}"
    return frame


@pytest.mark.slow
def test_calibrate_finds_homography_on_broadcast_footage():
    if not FIXTURE_VIDEO.exists():
        pytest.skip(f"fixture video not found: {FIXTURE_VIDEO}")
    if not (WEIGHTS_DIR / "SV_kp").exists():
        pytest.skip(
            "calibration weights not downloaded -- run scripts/download_calibration_weights.sh"
        )

    frame = _read_middle_frame(FIXTURE_VIDEO)
    calibrator = PitchCalibrator()
    result = calibrator.calibrate(frame)

    assert result.homography is not None
    assert result.homography.shape == (3, 3)
    # Phase 0 measured >=12 keypoints and >=1 line on every match_4 test frame
    assert result.num_keypoints >= 10
    assert result.top_kp_score > 0.5


@pytest.mark.slow
def test_calibrate_returns_none_homography_when_no_correspondences():
    """A blank frame has no pitch markings at all -- must not raise, must
    report no homography."""
    if not (WEIGHTS_DIR / "SV_kp").exists():
        pytest.skip(
            "calibration weights not downloaded -- run scripts/download_calibration_weights.sh"
        )

    import numpy as np

    blank = np.zeros((1080, 1920, 3), dtype="uint8")
    calibrator = PitchCalibrator()
    result = calibrator.calibrate(blank)

    assert result.homography is None


class _FakeCalibrator:
    """No real model -- exercises CalibrationWorker's threading/cadence logic
    only, deterministically and fast."""

    def __init__(self):
        self.calls = 0

    def calibrate(self, frame):
        self.calls += 1
        return CalibrationResult(
            homography=None,
            num_keypoints=0,
            num_lines=0,
            top_kp_score=0.0,
            top_line_score=0.0,
        )


def test_calibration_worker_updates_result_after_submit():
    import time

    from scripts.calibration import CalibrationWorker

    fake = _FakeCalibrator()
    worker = CalibrationWorker(fake, keyframe_interval=1).start()
    try:
        worker.submit(frame="fake_frame", frame_id=1)
        for _ in range(50):
            if fake.calls > 0:
                break
            time.sleep(0.01)
        assert fake.calls > 0
        assert worker.get_result() is not None
    finally:
        worker.stop()


def test_calibration_worker_respects_keyframe_interval():
    """Submitting frames faster than the keyframe interval must not
    calibrate every single one -- that's the whole point of a keyframe
    cadence."""
    import time

    from scripts.calibration import CalibrationWorker

    fake = _FakeCalibrator()
    worker = CalibrationWorker(fake, keyframe_interval=1000).start()
    try:
        for frame_id in range(1, 21):
            worker.submit(frame="fake_frame", frame_id=frame_id)
            time.sleep(0.005)
        time.sleep(0.1)
        assert fake.calls <= 1
    finally:
        worker.stop()
