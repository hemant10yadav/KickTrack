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


def _select_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class CalibrationResult:
    homography: np.ndarray | None
    num_keypoints: int
    num_lines: int
    top_kp_score: float
    top_line_score: float


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
            with self.lock:
                self.latest = result
                self.last_calibrated_frame_id = frame_id

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending
