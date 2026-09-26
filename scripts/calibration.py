"""Single-frame pitch calibration: turns a video frame into a homography
from pixel coordinates to real-world pitch coordinates (meters), using
PnLCalib's pretrained points+lines model (see docs/PITCH_CALIBRATION_SPEC.md
for why this model and why points+lines, not point-only).

Model inference takes 0.3-1.8s/frame (MPS vs CPU) -- this class only does
one frame at a time; CalibrationWorker (in this same module) is what runs it
at a sustainable keyframe cadence in the background.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
import yaml
from PIL import Image

from scripts.pnlcalib.model.cls_hrnet import get_cls_net
from scripts.pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from scripts.pnlcalib.utils.utils_calib import FramebyFrameCalib
from scripts.pnlcalib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)

WEIGHTS_DIR = Path("weights/pnlcalib")
CONFIG_DIR = Path(__file__).parent / "pnlcalib" / "config"

# Thresholds are PnLCalib's own defaults, tuned on SoccerNet broadcast
# footage -- Phase 0 verified these work on this project's broadcast-style
# footage (match_4/match_5) and that lowering them does NOT help on
# out-of-domain footage (match_3), it just lets bad correspondences through.
KP_THRESHOLD = 0.3434
LINE_THRESHOLD = 0.7867

# PitchCalibrator's model input size (H, W) -- used for the warmup frame in
# _calibration_process_main. NOTE: frames sent to the child process are
# deliberately NOT pre-resized to this to save IPC payload size -- tried
# that and it silently changed calibration results (see calibrate()'s
# docstring and CalibrationWorker.submit()'s comment for why).
MODEL_INPUT_HW = (540, 960)

# Standard pitch line markings (meters, centered at the pitch's own center
# spot per this module's world-coordinate convention -- see
# _homography_from_cam_params). Used by scripts/display.py's
# PitchOverlayRenderer to draw the calibrated pitch outline on screen.
_RAW_PITCH_LINES = [
    [[0.0, 54.16], [16.5, 54.16]],
    [[16.5, 13.84], [16.5, 54.16]],
    [[16.5, 13.84], [0.0, 13.84]],
    [[88.5, 54.16], [105.0, 54.16]],
    [[88.5, 13.84], [88.5, 54.16]],
    [[88.5, 13.84], [105.0, 13.84]],
    [[52.5, 0.0], [52.5, 68.0]],
    [[0.0, 68.0], [105.0, 68.0]],
    [[0.0, 0.0], [0.0, 68.0]],
    [[105.0, 0.0], [105.0, 68.0]],
    [[0.0, 0.0], [105.0, 0.0]],
    [[0.0, 43.16], [5.5, 43.16]],
    [[5.5, 43.16], [5.5, 24.84]],
    [[5.5, 24.84], [0.0, 24.84]],
    [[99.5, 43.16], [105.0, 43.16]],
    [[99.5, 43.16], [99.5, 24.84]],
    [[99.5, 24.84], [105.0, 24.84]],
]
PITCH_LINES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((x1 - 52.5, y1 - 34.0), (x2 - 52.5, y2 - 34.0)) for (x1, y1), (x2, y2) in _RAW_PITCH_LINES
]


def _select_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class CalibrationResult:
    homography: np.ndarray | None
    num_keypoints: int
    num_lines: int
    top_kp_score: float
    top_line_score: float
    frame_id: int | None = None


def _homography_from_cam_params(final_params_dict: dict) -> np.ndarray:
    """Reduces PnLCalib's full 3x4 camera projection matrix to a 3x3
    homography for the pitch plane (Z=0), by dropping the Z column of the
    projection matrix -- valid because every point this project cares about
    (player positions, pitch markings) lies on the pitch plane.
    """
    cam_params = final_params_dict["cam_params"]
    principal_point = np.array(cam_params["principal_point"])
    position_meters = np.array(cam_params["position_meters"])
    rotation = np.array(cam_params["rotation_matrix"])

    extrinsics = np.eye(4)[:-1]
    extrinsics[:, -1] = -position_meters
    intrinsics = np.array(
        [
            [cam_params["x_focal_length"], 0, principal_point[0]],
            [0, cam_params["y_focal_length"], principal_point[1]],
            [0, 0, 1],
        ]
    )
    projection = intrinsics @ (rotation @ extrinsics)  # 3x4: world[X,Y,Z,1] -> image
    return projection[:, [0, 1, 3]]  # drop Z column -> pitch-plane [X,Y,1] -> image


class PitchCalibrator:
    def __init__(self, device: str | None = None):
        self.device = device or _select_device()
        cfg = yaml.safe_load(open(CONFIG_DIR / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(CONFIG_DIR / "hrnetv2_w48_l.yaml"))

        self.model = get_cls_net(cfg)
        self.model.load_state_dict(torch.load(WEIGHTS_DIR / "SV_kp", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.model_l = get_cls_net_l(cfg_l)
        self.model_l.load_state_dict(torch.load(WEIGHTS_DIR / "SV_lines", map_location=self.device))
        self.model_l.to(self.device)
        self.model_l.eval()

        self._resize = T.Resize(MODEL_INPUT_HW)

    def calibrate(
        self, frame: np.ndarray, original_shape: tuple[int, int] | None = None
    ) -> CalibrationResult:
        """`original_shape` (h, w) lets a caller pass an already-downsized
        `frame` (CalibrationWorker does this to cut IPC payload size) while
        still getting a homography addressing the TRUE original pixel
        space -- FramebyFrameCalib denormalizes into whatever (w_orig,
        h_orig) it's told, regardless of what resolution the tensor
        actually was. Getting this wrong would silently produce a
        homography scaled to the wrong resolution.
        """
        h_orig, w_orig = original_shape if original_shape is not None else frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tvf.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)
        if tensor.size()[-1] != 960:
            tensor = self._resize(tensor)
        tensor = tensor.to(self.device)
        _, _, h, w = tensor.size()

        with torch.no_grad():
            heatmaps = self.model(tensor)
            heatmaps_l = self.model_l(tensor)

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

        top_kp_score = float(kp_coords[0, :, 0, 2].max()) if kp_coords.numel() else 0.0
        top_line_score = float(line_coords[0, :, :, 2].max()) if line_coords.numel() else 0.0

        kp_dict = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
        lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=w, h=h, normalize=True
        )

        cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
        cam.update(kp_dict, lines_dict)
        final_params_dict = cam.heuristic_voting(refine_lines=True)

        homography = _homography_from_cam_params(final_params_dict) if final_params_dict else None

        return CalibrationResult(
            homography=homography,
            num_keypoints=len(kp_dict),
            num_lines=len(lines_dict),
            top_kp_score=top_kp_score,
            top_line_score=top_line_score,
        )


CALIBRATOR_READY = "ready"  # the child's first message, once its model is warmed up


def _calibration_process_main(
    calibrator_factory: Callable[[], object],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    """Entry point for CalibrationWorker's child process. Constructs the
    calibrator here (not in the parent) so no torch model/device state ever
    needs to cross the process boundary -- only frames and results do.
    """
    # PyTorch defaults to spawning a CPU thread pool sized to a large chunk
    # of available cores (5 of 11 on this project's M3 Pro) -- fine for a
    # lone process, but it competed with the main process's real-time video
    # loop for actual CPU cores even after moving off its own thread
    # (measured: still ~70/502 dropped frames on match_5 with this
    # uncapped). Capping it here keeps this process's CPU footprint small.
    torch.set_num_threads(1)
    # Even with threads capped, this process's bursts of real CPU work
    # still measurably starved InferenceWorker's thread in the main process
    # on match_5's tight 50fps/20ms budget. Lowering this process's OS
    # scheduling priority is the theoretically-correct ask ("prefer the
    # real-time work when competing for a core") -- measured with A/B
    # testing, though, it made no detectable difference on its own (still
    # ~33-40/502 dropped either way). Left in as a harmless, standard
    # practice for background work; the actual fix was reducing how often
    # this process's bursts happen at all -- see keyframe_interval's
    # default and docs/PITCH_CALIBRATION_SPEC.md.
    try:
        os.nice(10)
    except (AttributeError, PermissionError, OSError):
        pass  # os.nice is POSIX-only and can be refused by the OS; not fatal either way
    calibrator = calibrator_factory()
    # Pay the one-time MPS kernel JIT-compilation cost now (measured: first
    # real call ~570ms vs ~420ms steady-state) rather than on the first real
    # keyframe -- same reasoning as PlayerTracker._warmup() for the YOLO
    # model. The model always resizes its input to MODEL_INPUT_HW
    # internally, so warmup shape doesn't need to match any real frame.
    calibrator.calibrate(np.zeros((*MODEL_INPUT_HW, 3), dtype=np.uint8))
    output_queue.put(CALIBRATOR_READY)
    while True:
        item = input_queue.get()
        if item is None:  # stop sentinel, see CalibrationWorker.stop()
            break
        frame, frame_id, original_shape = item
        result = calibrator.calibrate(frame, original_shape=original_shape)
        result.frame_id = frame_id
        output_queue.put(result)  # frame itself isn't sent back -- the
        # parent already has it (it's the one that submitted it); sending
        # a ~6MB frame both ways doubles pickling cost for no reason.


class CalibrationWorker:
    """Runs PitchCalibrator in a background *process* (not a thread) at a
    fixed keyframe cadence, never blocking the caller. Mirrors
    InferenceWorker's submit/get_result pattern (see scripts/pipeline.py)
    but only actually calibrates every `keyframe_interval`-th submitted
    frame_id -- calibration is too slow (0.3-1.8s) to run on every frame
    like detection does.

    A separate *process*, not a thread, is deliberate: measured (see
    docs/PITCH_CALIBRATION_SPEC.md Phase 2 GIL-contention investigation)
    that running this as a background thread let its heavy PyTorch
    computation starve InferenceWorker's own thread of Python's GIL,
    reintroducing frame drops on match_5's demanding 50fps footage (74/502
    dropped, vs 0 with calibration's heavy work moved off-thread). A
    separate process has no GIL to share, so it can't do that.
    """

    def __init__(
        self,
        calibrator_factory: Callable[[], object] = PitchCalibrator,
        # 30 (recalibrate ~every 0.6s at 50fps) measurably starved
        # InferenceWorker's thread even after process isolation + thread
        # capping (see docs/PITCH_CALIBRATION_SPEC.md): each full
        # recalibration is a real CPU burst, and drops scaled roughly
        # proportionally with how often it happens (30->~10.5%, 60->~5.3%,
        # 90->~4% dropped on match_5.mp4). 90 (~1.8s between recalibrations)
        # keeps InferenceWorker's drop rate consistently under the 5%
        # regression ceiling with real margin, while HomographyWorker's
        # per-frame propagation (not this) is what actually keeps the
        # homography smooth in between -- this interval only controls how
        # often it gets "trued up" against a fresh full calibration.
        keyframe_interval: int = 90,
    ):
        self.keyframe_interval = keyframe_interval
        self.input_queue: mp.Queue = mp.Queue(maxsize=1)
        self.output_queue: mp.Queue = mp.Queue()
        self.process = mp.Process(
            target=_calibration_process_main,
            args=(calibrator_factory, self.input_queue, self.output_queue),
            daemon=True,
        )
        self.lock = threading.Lock()
        self.latest: CalibrationResult | None = None
        self.latest_frame: np.ndarray | None = None
        self.last_calibrated_frame_id = None
        # Set by the drain thread; see is_keyframe() for why both gate submits.
        self.child_ready = False
        self.in_flight = False
        self.running = True
        self.drain_thread = threading.Thread(target=self._drain, daemon=True)
        # Frames submitted as keyframes, keyed by frame_id, so get_keyframe()
        # can pair a result with its frame without the child process having
        # to send the (large) frame back too -- the parent already has it.
        # Holds at most one entry: a frame is only submitted while none is
        # in flight (see is_keyframe()).
        self._submitted_frames: dict[int, np.ndarray] = {}

    def start(self) -> CalibrationWorker:
        self.process.start()
        self.drain_thread.start()
        return self

    def stop(self):
        self.running = False
        try:
            self.input_queue.put_nowait(None)  # stop sentinel
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
        self.drain_thread.join(timeout=2)
        # Without this, a multiprocessing.Queue's internal feeder thread can
        # block the *interpreter's own exit* trying to flush any leftover
        # unread item to the pipe (observed: a process hanging at
        # _Py_Finalize/atexit for minutes after the video had already
        # finished processing, stuck in ThreadHandle_join). Since a
        # leftover in-flight calibration frame is fine to simply drop on
        # shutdown, tell both queues not to bother.
        self.input_queue.cancel_join_thread()
        self.output_queue.cancel_join_thread()

    def submit(self, frame, frame_id: int):
        if not self.is_keyframe(frame_id):
            return
        # NOTE: deliberately NOT pre-resizing before sending to the child to
        # shrink the IPC payload -- tried that (cv2.resize on the raw uint8
        # frame) and it silently changed calibration results: the original
        # path resizes a *float* tensor via torchvision's Resize (which
        # anti-aliases), so pre-resizing with cv2.resize instead measurably
        # changed keypoint detections (14 -> 13 in one A/B test) and
        # produced a substantially different homography, not just an
        # equivalent lower-resolution one. Not worth the accuracy risk for
        # a performance win -- send the untouched frame.
        original_shape = frame.shape[:2]  # (h, w)
        try:
            self.input_queue.put_nowait((frame, frame_id, original_shape))
        except queue.Full:
            return
        with self.lock:
            self.in_flight = True
            self._submitted_frames[frame_id] = frame

    def get_result(self) -> CalibrationResult | None:
        with self.lock:
            return self.latest

    def get_keyframe(self) -> tuple[CalibrationResult, np.ndarray] | None:
        """Returns the latest calibration result together with the exact
        frame it was computed from, atomically -- so a caller can tell
        whether this is a *new* keyframe (compare `.frame_id`) and, if so,
        reset a HomographyPropagator against the matching frame."""
        with self.lock:
            if self.latest is None:
                return None
            return self.latest, self.latest_frame

    def is_keyframe(self, frame_id: int) -> bool:
        """Public so callers can skip work (e.g. copying a frame) that's
        only needed when this frame_id will actually be submitted -- see
        PlayerTracker._play(), which used to copy every single frame for
        this worker even though ~29/30 of those copies were immediately
        discarded by submit()'s own internal check."""
        # Only an idle child gets a frame. This used to hand one over on every
        # due frame and pull the previous, still-unstarted one back out of the
        # queue ("latest wins") -- unpickling a whole frame on the display
        # thread, on every frame of a 0.3-1.8s calibration: ~26ms per 4K
        # frame (~6ms at 1080p), 156 of the first 300 frames of a 4K match_5.
        # An idle child starts on the frame at once, so it is just as fresh.
        if not self.child_ready or self.in_flight:
            return False
        if self.last_calibrated_frame_id is None:
            return True
        return frame_id - self.last_calibrated_frame_id >= self.keyframe_interval

    def _drain(self):
        """Runs in a lightweight thread (not the child process) purely to
        move finished results out of the multiprocessing queue and into
        self.latest without making get_result()/get_keyframe() block."""
        while self.running:
            try:
                result = self.output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(result, str) and result == CALIBRATOR_READY:
                self.child_ready = True
                continue
            with self.lock:
                self.latest = result
                self.latest_frame = self._submitted_frames.pop(result.frame_id, None)
                self.last_calibrated_frame_id = result.frame_id
                self.in_flight = False


MIN_TRACKED_POINTS = (
    15  # below this, propagation is unreliable -- caller should hold last homography
)
MAX_TRACK_ERROR = (
    10.0  # cv2 LK tracking error above this means "not really tracked" (see propagate())
)


# Anchored propagation (docs/PLAN.md Plan 2.11). Measured on match_5 against a
# full calibration of every 10th frame (keyframe every 90 frames): chaining
# frame-to-frame homographies at a 3px RANSAC threshold drifted 20px median,
# 73px p90, 106px max by the next keyframe. A pan moves the picture 1-3px per
# frame, inside that threshold, so points that do not move with the pitch
# (the broadcast scoreboard, players) passed as inliers and pulled every step
# toward "no motion". Estimating each frame's motion from an anchor up to
# ANCHOR_FRAMES back at a 1px threshold: 6.7px median, 14px p90, 26px max --
# the calibrations themselves differ ~10px from one to the next.
ANCHOR_FRAMES = 10
RANSAC_THRESHOLD_PX = 1.0
# Corners are taken a few per cell of a grid, not the strongest N of the whole
# frame: on match_4 those were nearly all on the broadcast's static graphics
# (ticker text, scoreboard, logo), whose corners are far sharper than crowd or
# grass, so 103 of 111 tracked points agreed on "no motion" while the camera
# panned 30px (frames 1400-1410) and the overlay ran up to ~1200px off. Grid,
# match_4: >50px frames 203 -> 4 of 849 checked; match_5 unchanged.
# A blurred frame in a fast pan can leave too few points within 1px; retrying
# at 3px (match_4 frames 1449-1555) instead of giving up let propagation keep
# up rather than freeze until the next keyframe: max 828 -> 49px.
RETRY_THRESHOLD_PX = 3.0
CORNER_GRID = (8, 5)  # columns, rows
CORNERS_PER_CELL = 5


def landmark_distance(homography_a: np.ndarray, homography_b: np.ndarray, frame_shape) -> float:
    """Median pixel distance between where two pitch -> image homographies put
    a grid of pitch points, over the points `homography_b` puts in frame."""
    h, w = frame_shape[:2]
    xs, ys = np.meshgrid(np.linspace(-52.5, 52.5, 15), np.linspace(-34.0, 34.0, 11))
    pitch = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])

    def project(homography):
        points = homography @ pitch
        with np.errstate(divide="ignore", invalid="ignore"):
            return points[:2] / points[2]

    a, b = project(homography_a), project(homography_b)
    inside = (b[0] >= 0) & (b[0] < w) & (b[1] >= 0) & (b[1] < h)
    if not inside.any():
        return float("inf")
    return float(np.median(np.linalg.norm(a[:, inside] - b[:, inside], axis=0)))


def _grid_corners(gray: np.ndarray) -> np.ndarray | None:
    """Up to CORNERS_PER_CELL corners from each CORNER_GRID cell (see
    CORNER_GRID), found on the frame at half size -- 10.7 -> 3.2ms per anchor
    at 1080p, drift unchanged; optical flow still tracks them at full size."""
    gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    columns, rows = CORNER_GRID
    found = []
    for row in range(rows):
        for column in range(columns):
            y0, x0 = row * h // rows, column * w // columns
            cell = gray[y0 : (row + 1) * h // rows, x0 : (column + 1) * w // columns]
            corners = cv2.goodFeaturesToTrack(
                cell, maxCorners=CORNERS_PER_CELL, qualityLevel=0.01, minDistance=10
            )
            if corners is not None:
                found.append(corners + np.array([x0, y0], dtype=np.float32))
    return np.concatenate(found) * 2 if found else None


class HomographyPropagator:
    """Tracks the pitch homography frame-to-frame between full
    CalibrationWorker recalibrations, using sparse optical flow. Conceptually
    mirrors ResultTimeline's role (scripts/display.py) of turning a
    background worker's keyframe-rate results into a per-frame estimate --
    but unlike ResultTimeline (cheap enough to run inline), this runs on
    HomographyWorker's background thread (below), not the main thread
    directly: see HomographyWorker's docstring for why.

    Each frame's camera motion is measured from an *anchor* frame up to
    ANCHOR_FRAMES back, not from the previous frame: ten 1-frame hops of
    1-3px each are where both the error accumulation and the bias toward
    static screen overlays came from (see ANCHOR_FRAMES). Every
    ANCHOR_FRAMES the current frame becomes the anchor, with fresh corners.

    If tracking ever fails (too few inlier points -- fast pan, occlusion,
    a bad frame), propagate() returns None and keeps measuring against the
    *same* anchor until it succeeds again or the next keyframe arrives,
    rather than guessing from a potentially-bad current frame. Callers
    should treat a None as "hold the last known-good homography, mark this
    frame low-confidence."
    """

    def __init__(self, min_tracked_points: int = MIN_TRACKED_POINTS):
        self.min_tracked_points = min_tracked_points
        self.anchor_gray: np.ndarray | None = None
        self.anchor_points: np.ndarray | None = None
        self.anchor_homography: np.ndarray | None = None
        self.homography: np.ndarray | None = None
        self._last_tracked: np.ndarray | None = None  # anchor points' latest positions
        self._frames_since_anchor = 0

    def reset(self, frame: np.ndarray, homography: np.ndarray):
        self._anchor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), homography)
        self.homography = homography

    def rebase(self, correction: np.ndarray):
        """Right-multiplies a pitch-side correction into every homography
        held, so a new keyframe calibration takes effect from *now* instead
        of restarting propagation from its (0.3-1.8s old) frame -- see
        HomographyWorker._apply_keyframe."""
        self.anchor_homography = self.anchor_homography @ correction
        self.homography = self.homography @ correction

    def propagate(self, frame: np.ndarray) -> np.ndarray | None:
        if self.homography is None or self.anchor_points is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked = self._track(gray)
        if tracked is None:
            return None

        transform, self._last_tracked = tracked
        self.homography = transform @ self.anchor_homography
        self._frames_since_anchor += 1
        if self._frames_since_anchor >= ANCHOR_FRAMES:
            self._anchor(gray, self.homography)
        return self.homography

    def _track(self, gray: np.ndarray):
        """(anchor -> gray transform, tracked points), or None."""
        if self.anchor_points is None:
            return None
        tracked_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray,
            gray,
            self.anchor_points,
            self._last_tracked.copy(),  # start from where they were last seen
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        # cv2 can report status=1 ("tracked") even when the target region has
        # no real content to track against (e.g. a blank/black frame) -- its
        # own per-point tracking error is what actually distinguishes that
        # case (measured ~127 on a blank frame vs ~0.0003 on real motion).
        status_mask = status.reshape(-1).astype(bool) & (err.reshape(-1) < MAX_TRACK_ERROR)
        if status_mask.sum() < self.min_tracked_points:
            return None
        for threshold in (RANSAC_THRESHOLD_PX, RETRY_THRESHOLD_PX):
            transform, inliers = cv2.findHomography(
                self.anchor_points[status_mask], tracked_points[status_mask], cv2.RANSAC, threshold
            )
            if transform is not None and int(inliers.sum()) >= self.min_tracked_points:
                return transform, tracked_points
        return None

    def _anchor(self, gray: np.ndarray, homography: np.ndarray):
        self.anchor_gray = gray
        self.anchor_points = _grid_corners(gray)
        self.anchor_homography = homography
        self._last_tracked = None if self.anchor_points is None else self.anchor_points.copy()
        self._frames_since_anchor = 0


class HomographyWorker:
    """Runs HomographyPropagator in a background *thread* (not a process --
    unlike CalibrationWorker, whose PyTorch work doesn't release the GIL,
    HomographyPropagator's calls are OpenCV C++ that do release it during
    the heavy computation, so a thread doesn't starve other threads the way
    the calibration model did), so the main display thread never blocks on
    optical-flow tracking.

    Measured why this matters (docs/PITCH_CALIBRATION_SPEC.md Phase 2):
    propagate() running synchronously in the main thread on match_5 (50fps,
    20ms/frame budget) pushed ~9-10% of frames over budget even though
    propagate() itself only costs ~1-3ms -- on a budget that thin, any
    added synchronous cost is risky. Async removes it from the budget
    entirely, regardless of its own cost.

    The propagator instance is owned exclusively by this worker's thread --
    reset() and submit() calls from other threads only ever enqueue a
    request (single-slot, "latest wins", with reset always taking priority
    over a pending propagate so a new keyframe is never dropped in favor of
    a stale propagate request), never touch the propagator directly.

    A keyframe calibration describes a frame that is already 0.3-1.8s old
    when it arrives. It is applied as a correction on top of what propagation
    had for that same frame (see _apply_keyframe), not by restarting from it:
    restarting snapped the overlay back to where the camera had been and then
    caught up again (docs/PLAN.md Plan 3.1, "double-snaps"). A keyframe that
    lands far from propagation is held back as a likely calibration glitch;
    if the next keyframe lands in the same place, propagation was the one
    that had drifted and both are right.
    """

    HISTORY_FRAMES = 256  # > the longest calibration latency (~1.8s = 90 frames at 50fps)
    MAX_KEYFRAME_GAP = 2  # frames between the keyframe and the nearest propagated one
    # A keyframe disagreeing with propagation by more than this share of the
    # frame width is held back once. match_5: glitched calibrations landed
    # 205-287px (1080p) from their neighbours; propagation drift peaked at 26px.
    GLITCH_FRACTION_OF_WIDTH = 0.03

    def __init__(self, propagator: HomographyPropagator):
        self.propagator = propagator
        self.lock = threading.Lock()
        self.pending: tuple | None = None
        self.latest_homography: np.ndarray | None = None
        self.latest_frame_id: int | None = None
        self.history: deque[tuple[int, np.ndarray]] = deque(maxlen=self.HISTORY_FRAMES)
        self.held_correction: np.ndarray | None = None  # the last held-back keyframe's
        self.keyframes_rebased = 0
        self.keyframes_reset = 0
        self.keyframes_rejected = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> HomographyWorker:
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def reset(self, frame: np.ndarray, homography: np.ndarray, frame_id: int):
        """Call when a new keyframe calibration arrives."""
        with self.lock:
            self.pending = ("reset", frame, homography, frame_id)

    def submit(self, frame: np.ndarray, frame_id: int):
        """Call every displayed frame to request a propagated homography."""
        with self.lock:
            if self.pending is not None and self.pending[0] == "reset":
                return  # never drop a pending reset in favor of a propagate
            self.pending = ("propagate", frame, frame_id)

    def get_latest(self) -> tuple[np.ndarray | None, int | None]:
        with self.lock:
            return self.latest_homography, self.latest_frame_id

    def _take_pending(self) -> tuple | None:
        with self.lock:
            pending, self.pending = self.pending, None
            return pending

    def _run(self):
        while self.running:
            pending = self._take_pending()
            if pending is None:
                time.sleep(0.001)
                continue
            if pending[0] == "reset":
                _, frame, homography, frame_id = pending
                self._apply_keyframe(frame, homography, frame_id)
            else:
                _, frame, frame_id = pending
                homography = self.propagator.propagate(frame)
                if homography is not None:
                    self.history.append((frame_id, homography))
                    with self.lock:
                        self.latest_homography = homography
                        self.latest_frame_id = frame_id
                # else: hold whatever is already in self.latest_homography

    def summary(self) -> str:
        return (
            f"Keyframes applied on top of propagation: {self.keyframes_rebased}; "
            f"restarted from (no propagation for their frame): {self.keyframes_reset}; "
            f"held back as calibration glitches: {self.keyframes_rejected}"
        )

    def _apply_keyframe(self, frame: np.ndarray, homography: np.ndarray, frame_id: int):
        propagated = self._propagated_at(frame_id)
        if propagated is None or self.propagator.homography is None:
            self.keyframes_reset += 1
            self.history.clear()
            self.propagator.reset(frame, homography)
            with self.lock:
                self.latest_homography = homography
                self.latest_frame_id = frame_id
            return

        # propagated(now) = motion(k -> now) @ propagated(k), so the keyframe's
        # pitch-side correction carries forward unchanged:
        # calibrated(k) = propagated(k) @ correction.
        correction = np.linalg.inv(propagated) @ homography
        max_px = self.GLITCH_FRACTION_OF_WIDTH * frame.shape[1]
        now = self.propagator.homography
        if landmark_distance(now, now @ correction, frame.shape) > max_px:
            agrees_with_held = self.held_correction is not None and (
                landmark_distance(now @ self.held_correction, now @ correction, frame.shape)
                <= max_px
            )
            if not agrees_with_held:
                self.keyframes_rejected += 1
                self.held_correction = correction
                return
        self.held_correction = None
        self.keyframes_rebased += 1
        self.propagator.rebase(correction)
        self.history = deque(
            ((fid, h @ correction) for fid, h in self.history), maxlen=self.HISTORY_FRAMES
        )
        with self.lock:
            self.latest_homography = self.propagator.homography

    def _propagated_at(self, frame_id: int) -> np.ndarray | None:
        for history_frame_id, homography in reversed(self.history):
            if history_frame_id <= frame_id:
                if frame_id - history_frame_id <= self.MAX_KEYFRAME_GAP:
                    return homography
                return None
        return None
