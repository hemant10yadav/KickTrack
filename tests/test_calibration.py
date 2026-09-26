"""Tests for PitchCalibrator -- single-frame pixel-to-pitch homography.

Uses tests/videos/match_4.mp4 (broadcast-style footage that Phase 0 verified
calibrates reliably -- see docs/PITCH_CALIBRATION_SPEC.md). Slow (loads a
~270MB model, runs real inference), like tests/test_tracking_quality.py.
"""

from pathlib import Path

import cv2
import numpy as np
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
    """No real model -- exercises CalibrationWorker's process/cadence logic
    only, deterministically and fast. calibrate() runs inside the worker's
    *child process*, so call counts are tracked via a multiprocessing.Value
    (a real shared counter, not a plain attribute the parent couldn't see
    across the process boundary)."""

    def __init__(self, call_counter):
        self.call_counter = call_counter

    def calibrate(self, frame, original_shape=None):
        with self.call_counter.get_lock():
            self.call_counter.value += 1
        return CalibrationResult(
            homography=None,
            num_keypoints=0,
            num_lines=0,
            top_kp_score=0.0,
            top_line_score=0.0,
        )


def _make_fake_calibrator(call_counter):
    """Top-level (picklable) factory -- CalibrationWorker calls this with no
    arguments inside the child process, so the counter must be bound via
    functools.partial rather than passed as a normal call argument."""
    return _FakeCalibrator(call_counter)


def _wait_until_ready(worker, timeout_s=10.0):
    """Frames are only accepted once the child has warmed up its model (and
    while no other frame is in flight), so wait for that before submitting."""
    import time

    deadline = time.monotonic() + timeout_s
    while not worker.child_ready and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.child_ready


def test_calibration_worker_updates_result_after_submit():
    import functools
    import multiprocessing as mp
    import time

    from scripts.calibration import CalibrationWorker

    call_counter = mp.Value("i", 0)
    worker = CalibrationWorker(
        functools.partial(_make_fake_calibrator, call_counter), keyframe_interval=1
    ).start()
    try:
        _wait_until_ready(worker)
        worker.submit(frame=np.zeros((4, 4, 3), dtype="uint8"), frame_id=1)
        for _ in range(200):
            if worker.get_result() is not None:
                break
            time.sleep(0.05)
        assert call_counter.value == 2  # the child's warm-up call, then frame 1
        result = worker.get_result()
        assert result is not None
        assert result.frame_id == 1
    finally:
        worker.stop()


def test_calibration_worker_respects_keyframe_interval():
    """Submitting frames faster than the keyframe interval must not
    calibrate every single one -- that's the whole point of a keyframe
    cadence."""
    import functools
    import multiprocessing as mp
    import time

    from scripts.calibration import CalibrationWorker

    call_counter = mp.Value("i", 0)
    worker = CalibrationWorker(
        functools.partial(_make_fake_calibrator, call_counter), keyframe_interval=1000
    ).start()
    try:
        _wait_until_ready(worker)
        for frame_id in range(1, 21):
            worker.submit(frame=np.zeros((4, 4, 3), dtype="uint8"), frame_id=frame_id)
            time.sleep(0.005)
        time.sleep(0.5)
        assert call_counter.value == 2  # the child's warm-up call, then frame 1 only
    finally:
        worker.stop()


def test_calibration_worker_get_keyframe_returns_matching_frame():
    """Phase 2's HomographyPropagator needs the exact frame a calibration
    came from (to seed optical flow tracking against it), not just the
    result -- get_keyframe() must return both together, atomically."""
    import functools
    import multiprocessing as mp
    import time

    from scripts.calibration import CalibrationWorker

    call_counter = mp.Value("i", 0)
    worker = CalibrationWorker(
        functools.partial(_make_fake_calibrator, call_counter), keyframe_interval=1
    ).start()
    try:
        submitted_frame = np.zeros((4, 4, 3), dtype="uint8")
        submitted_frame[:] = 7
        _wait_until_ready(worker)
        worker.submit(frame=submitted_frame, frame_id=1)
        for _ in range(200):
            if worker.get_keyframe() is not None:
                break
            time.sleep(0.05)
        keyframe = worker.get_keyframe()
        assert keyframe is not None
        result, frame = keyframe
        assert result.frame_id == 1
        assert np.array_equal(frame, submitted_frame)
    finally:
        worker.stop()


def _make_textured_frame(size=300):
    """A synthetic checkerboard has strong, well-distributed corners for
    cv2.goodFeaturesToTrack/optical flow -- unlike a blank frame."""
    import numpy as np

    square = 20
    rows, cols = np.indices((size, size))
    grid = ((rows // square) + (cols // square)) % 2
    gray = (grid * 255).astype("uint8")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_homography_propagator_recovers_known_translation():
    """Warp a textured frame by a known pixel translation, seed the
    propagator at identity homography, and check it recovers that exact
    translation as the new homography -- proves the optical-flow-tracking
    + composition math is correct before trusting it on real footage."""
    from scripts.calibration import HomographyPropagator

    frame0 = _make_textured_frame()
    dx, dy = 8.0, -5.0
    translation = np.array([[1, 0, dx], [0, 1, dy]], dtype="float32")
    frame1 = cv2.warpAffine(frame0, translation, (frame0.shape[1], frame0.shape[0]))

    propagator = HomographyPropagator()
    propagator.reset(frame0, homography=np.eye(3))
    result = propagator.propagate(frame1)

    assert result is not None
    result = result / result[2, 2]
    expected = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    assert np.allclose(result, expected, atol=1.0)


def test_homography_propagator_fails_gracefully_on_blank_frame():
    """A blank frame has no trackable features at all -- must return None,
    not raise or silently produce a garbage homography."""
    from scripts.calibration import HomographyPropagator

    frame0 = _make_textured_frame()
    blank = np.zeros_like(frame0)

    propagator = HomographyPropagator()
    propagator.reset(frame0, homography=np.eye(3))
    result = propagator.propagate(blank)

    assert result is None


def _wait_until(predicate, timeout_s=5.0, interval_s=0.02):
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def test_homography_worker_reset_sets_latest_immediately():
    from scripts.calibration import HomographyPropagator, HomographyWorker

    frame0 = _make_textured_frame()
    worker = HomographyWorker(HomographyPropagator()).start()
    try:
        worker.reset(frame0, homography=np.eye(3), frame_id=5)
        assert _wait_until(lambda: worker.get_latest()[1] == 5)
        homography, frame_id = worker.get_latest()
        assert frame_id == 5
        assert np.array_equal(homography, np.eye(3))
    finally:
        worker.stop()


def test_homography_worker_submit_propagates_asynchronously():
    """Mirrors test_homography_propagator_recovers_known_translation but
    through the worker's async submit()/get_latest() -- proves the thread
    plumbing doesn't lose or corrupt the result, not just the underlying
    propagator math (already covered above)."""
    from scripts.calibration import HomographyPropagator, HomographyWorker

    frame0 = _make_textured_frame()
    dx, dy = 8.0, -5.0
    translation = np.array([[1, 0, dx], [0, 1, dy]], dtype="float32")
    frame1 = cv2.warpAffine(frame0, translation, (frame0.shape[1], frame0.shape[0]))

    worker = HomographyWorker(HomographyPropagator()).start()
    try:
        worker.reset(frame0, homography=np.eye(3), frame_id=1)
        assert _wait_until(lambda: worker.get_latest()[1] == 1)

        worker.submit(frame1, frame_id=2)
        assert _wait_until(lambda: worker.get_latest()[1] == 2)

        homography, frame_id = worker.get_latest()
        assert frame_id == 2
        homography = homography / homography[2, 2]
        expected = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        assert np.allclose(homography, expected, atol=1.0)
    finally:
        worker.stop()


def test_homography_worker_holds_last_good_when_propagate_fails():
    """A frame with nothing trackable (blank) must not erase the last
    known-good homography -- see HomographyPropagator's own
    hold-last-good contract, which the worker must preserve."""
    from scripts.calibration import HomographyPropagator, HomographyWorker

    frame0 = _make_textured_frame()
    blank = np.zeros_like(frame0)

    worker = HomographyWorker(HomographyPropagator()).start()
    try:
        worker.reset(frame0, homography=np.eye(3), frame_id=1)
        assert _wait_until(lambda: worker.get_latest()[1] == 1)

        worker.submit(blank, frame_id=2)
        import time

        time.sleep(0.3)  # let the worker actually attempt (and fail) propagation

        homography, frame_id = worker.get_latest()
        assert frame_id == 1  # unchanged -- frame 2's failed propagate must not overwrite it
        assert np.array_equal(homography, np.eye(3))
    finally:
        worker.stop()
