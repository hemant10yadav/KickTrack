"""Single-frame pitch calibration: turns a video frame into a homography
from pixel coordinates to real-world pitch coordinates (meters), using
PnLCalib's pretrained points+lines model (see docs/PITCH_CALIBRATION_SPEC.md
for why this model and why points+lines, not point-only).

Model inference takes 0.3-1.8s/frame (MPS vs CPU) -- this class only does
one frame at a time; CalibrationWorker (in this same module) is what runs it
at a sustainable keyframe cadence in the background.
"""

from __future__ import annotations

import threading
import time
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

        self._resize = T.Resize((540, 960))

    def calibrate(self, frame: np.ndarray) -> CalibrationResult:
        h_orig, w_orig = frame.shape[:2]
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


class CalibrationWorker:
    """Runs PitchCalibrator in the background at a fixed keyframe cadence,
    never blocking the caller. Mirrors InferenceWorker's submit/get_result
    pattern (see scripts/pipeline.py) but only actually calibrates every
    `keyframe_interval`-th submitted frame_id -- calibration is too slow
    (0.3-1.8s) to run on every frame like detection does.
    """

    def __init__(self, calibrator: PitchCalibrator, keyframe_interval: int = 30):
        self.calibrator = calibrator
        self.keyframe_interval = keyframe_interval
        self.lock = threading.Lock()
        self.pending = None  # (frame, frame_id)
        self.latest: CalibrationResult | None = None
        self.latest_frame: np.ndarray | None = None
        self.last_calibrated_frame_id = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> CalibrationWorker:
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def submit(self, frame, frame_id: int):
        if not self._is_keyframe(frame_id):
            return
        with self.lock:
            self.pending = (frame, frame_id)

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

    def _is_keyframe(self, frame_id: int) -> bool:
        if self.last_calibrated_frame_id is None:
            return True
        return frame_id - self.last_calibrated_frame_id >= self.keyframe_interval

    def _run(self):
        while self.running:
            pending = self._take_pending()
            if pending is None:
                time.sleep(0.005)
                continue
            frame, frame_id = pending
            result = self.calibrator.calibrate(frame)
            result.frame_id = frame_id
            with self.lock:
                self.latest = result
                self.latest_frame = frame
                self.last_calibrated_frame_id = frame_id

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending


MIN_TRACKED_POINTS = (
    15  # below this, propagation is unreliable -- caller should hold last homography
)
MAX_TRACK_ERROR = (
    10.0  # cv2 LK tracking error above this means "not really tracked" (see propagate())
)


class HomographyPropagator:
    """Tracks the pitch homography frame-to-frame between full
    CalibrationWorker recalibrations, using sparse optical flow. Mirrors
    MotionExtrapolator's role (scripts/display.py) of running every frame in
    the main thread on top of a background worker's keyframe-rate results.

    See docs/PITCH_CALIBRATION_SPEC.md Phase 2 for the drift measurement
    that motivated this: match_4-style footage (more camera movement)
    drifts ~125px on average by the time the current 30-frame keyframe
    interval elapses if the homography is just held stale.

    If tracking ever fails (too few inlier points -- fast pan, occlusion,
    a bad frame), propagate() returns None and keeps failing on the *same*
    stale reference until the next full keyframe recalibration resets it,
    rather than guessing from a potentially-bad current frame. Callers
    should treat a None as "hold the last known-good homography, mark this
    frame low-confidence."
    """

    def __init__(self, max_corners: int = 200, min_tracked_points: int = MIN_TRACKED_POINTS):
        self.max_corners = max_corners
        self.min_tracked_points = min_tracked_points
        self.reference_gray: np.ndarray | None = None
        self.reference_points: np.ndarray | None = None
        self.homography: np.ndarray | None = None

    def reset(self, frame: np.ndarray, homography: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = gray
        self.reference_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=10
        )
        self.homography = homography

    def propagate(self, frame: np.ndarray) -> np.ndarray | None:
        if self.homography is None or self.reference_points is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.reference_gray, gray, self.reference_points, None
        )
        # cv2 can report status=1 ("tracked") even when the target region has
        # no real content to track against (e.g. a blank/black frame) -- its
        # own per-point tracking error is what actually distinguishes that
        # case (measured ~127 on a blank frame vs ~0.0003 on real motion).
        status_mask = status.reshape(-1).astype(bool) & (err.reshape(-1) < MAX_TRACK_ERROR)
        if status_mask.sum() < self.min_tracked_points:
            return None

        src = self.reference_points[status_mask]
        dst = tracked_points[status_mask]
        transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if transform is None or int(inliers.sum()) < self.min_tracked_points:
            return None

        new_homography = transform @ self.homography

        # Roll the reference forward to the current frame so optical flow
        # always tracks a short hop from the previous frame, not an
        # ever-growing distance from the original keyframe.
        inlier_mask = inliers.reshape(-1).astype(bool)
        self.reference_gray = gray
        self.reference_points = dst[inlier_mask]
        self.homography = new_homography
        return new_homography
