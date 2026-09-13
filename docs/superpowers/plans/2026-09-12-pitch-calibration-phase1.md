# Pitch Calibration Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire keyframe-rate pitch calibration (pixel -> real-world pitch coordinates)
into the tracking pipeline as a background worker, using PnLCalib's pretrained
points+lines model, validated in Phase 0 against this project's broadcast-style
footage (`match_4`/`match_5`).

**Architecture:** Vendor PnLCalib's inference-only code into `scripts/pnlcalib/`
(no training/eval scripts). Add `scripts/calibration.py` with a `PitchCalibrator`
(single-frame calibration, wraps the vendored model) and a `CalibrationWorker`
(background thread, same submit/get_result pattern as the existing
`InferenceWorker` in `scripts/pipeline.py`) that calibrates on the latest frame
at a fixed cadence (not every frame — inference takes 0.3-1.8s). Wire the worker
into `PlayerTracker` alongside the existing `InferenceWorker`. This phase does
NOT add frame-to-frame propagation (Phase 2) or PlayerState/pixel-to-pitch
coordinate conversion for other consumers (Phase 3) — it only gets a homography
computed and available, at keyframe rate, inside the running pipeline.

**Tech Stack:** PyTorch/torchvision (CPU or MPS), PnLCalib's HRNet-based
keypoint/line model (vendored, not pip-installed), the project's existing
threading-based worker pattern.

**Spec:** `docs/PITCH_CALIBRATION_SPEC.md` (Phase 0 results + Phase 1 scope)

## Global Constraints

- Model inference is 0.3-1.8s/frame (measured: MPS ~0.33-0.41s, CPU ~1.6-1.8s on
  this project's M3 Pro) — never call `PitchCalibrator.calibrate()` from a
  real-time loop directly; only from `CalibrationWorker`'s background thread.
- Weights (`SV_kp`, `SV_lines`) are gitignored, like `yolov8m.mlpackage` and
  other model weights (see `CLAUDE.md` file layout) — never committed.
- Vendored PnLCalib code is inference-only (`model/`, `utils/`, `config/`
  subset) — no training scripts, no eval scripts, no dataset-loading code.
- Follow the project's "group files by domain" convention: calibration code
  lives in one `scripts/calibration.py`, not split into one-class-per-file.
- This phase does not touch `scripts/player.py` or `scripts/display.py`.

---

### Task 1: Vendor PnLCalib inference code and add weight download

**Files:**
- Create: `scripts/pnlcalib/__init__.py` (empty)
- Create: `scripts/pnlcalib/model/__init__.py`, `scripts/pnlcalib/model/cls_hrnet.py`,
  `scripts/pnlcalib/model/cls_hrnet_l.py`, and any modules those two files import
  from `model/` in the upstream repo (check their own imports and copy
  transitively — e.g. shared blocks/config modules under `model/`)
- Create: `scripts/pnlcalib/utils/__init__.py`, `scripts/pnlcalib/utils/utils_calib.py`,
  `scripts/pnlcalib/utils/utils_heatmap.py`
- Create: `scripts/pnlcalib/config/hrnetv2_w48.yaml`, `scripts/pnlcalib/config/hrnetv2_w48_l.yaml`
- Create: `scripts/pnlcalib/LICENSE_NOTICE.md` (one line: source repo + license,
  see below)
- Create: `scripts/download_calibration_weights.sh`
- Modify: `.gitignore`
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: importable `scripts.pnlcalib.model.cls_hrnet.get_cls_net(cfg)`,
  `scripts.pnlcalib.model.cls_hrnet_l.get_cls_net(cfg)` (returns the line
  model despite the same function name — matches upstream naming, imported
  with an alias by Task 2), `scripts.pnlcalib.utils.utils_calib.FramebyFrameCalib`,
  `scripts.pnlcalib.utils.utils_heatmap.{get_keypoints_from_heatmap_batch_maxpool,
  get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict}`

- [ ] **Step 1: Copy the vendored source files**

Copy from the already-cloned prototype checkout used in Phase 0
(`mguti97/PnLCalib` at the commit used for Phase 0 testing) into
`scripts/pnlcalib/`:
- `model/cls_hrnet.py`, `model/cls_hrnet_l.py`, plus every file under `model/`
  that those two import (open each file, follow `from model.X import Y` /
  `from .X import Y` lines, copy `X.py` too, repeat until no new local
  imports remain)
- `utils/utils_calib.py`, `utils/utils_heatmap.py` (same transitive-import
  check — do NOT copy `utils/utils_display.py` or other files unrelated to
  single-frame calibration if present)
- `config/hrnetv2_w48.yaml`, `config/hrnetv2_w48_l.yaml`

Do NOT copy: `train.py`, `train_l.py`, `eval_wp.py`, `sn_calibration/`,
anything under `scripts/` (upstream's own scripts dir, not this project's),
or dataset-loading code.

Add `scripts/pnlcalib/LICENSE_NOTICE.md`:
```markdown
Vendored from https://github.com/mguti97/PnLCalib (inference-only subset:
model/, utils/, config/). See upstream repository for license terms.
```

- [ ] **Step 2: Verify the vendored code imports cleanly**

Run: `uv run python -c "from scripts.pnlcalib.model.cls_hrnet import get_cls_net; from scripts.pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l; from scripts.pnlcalib.utils.utils_calib import FramebyFrameCalib; from scripts.pnlcalib.utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict; print('ok')"`

Expected: prints `ok` with no `ImportError`/`ModuleNotFoundError`. If a
transitive import is missing, copy that file too and re-run.

- [ ] **Step 3: Add the weight download script**

```bash
#!/usr/bin/env bash
# Downloads PnLCalib's pretrained single-view keypoint/line detection
# weights (SoccerNet-pretrained). Gitignored -- see CLAUDE.md known
# gotchas: GitHub release downloads can be slow/unstable on this network.
set -euo pipefail
mkdir -p weights/pnlcalib
curl -L -o weights/pnlcalib/SV_kp \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o weights/pnlcalib/SV_lines \
  https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
echo "Downloaded weights/pnlcalib/{SV_kp,SV_lines}"
```

Save as `scripts/download_calibration_weights.sh`, `chmod +x` it.

- [ ] **Step 4: Add `weights/` to `.gitignore`**

Add this line under the existing "Model weights" section in `.gitignore`:
```
weights/pnlcalib/
```

- [ ] **Step 5: Move calibration dependencies into main project dependencies**

Run: `uv remove --group pnlcalib-prototype lsq-ellipse scipy shapely`
Run: `uv add lsq-ellipse scipy shapely`

(`torch`/`torchvision` are already main dependencies transitively via
`ultralytics` — no change needed there. This moves the Phase-0-only
prototype group's additions into the project's real dependencies now that
Phase 1 uses them in production code, not just a throwaway script.)

- [ ] **Step 6: Run the weight download and verify files exist**

Run: `./scripts/download_calibration_weights.sh`
Expected: `weights/pnlcalib/SV_kp` and `weights/pnlcalib/SV_lines` exist,
each roughly 250-270MB.

- [ ] **Step 7: Commit**

```bash
git add scripts/pnlcalib/ scripts/download_calibration_weights.sh .gitignore pyproject.toml uv.lock
git commit -m "Vendor PnLCalib inference code for pitch calibration"
```

---

### Task 2: `PitchCalibrator` — single-frame calibration

**Files:**
- Create: `scripts/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `scripts.pnlcalib.model.cls_hrnet.get_cls_net`,
  `scripts.pnlcalib.model.cls_hrnet_l.get_cls_net` (aliased), 
  `scripts.pnlcalib.utils.utils_calib.FramebyFrameCalib`,
  `scripts.pnlcalib.utils.utils_heatmap.*` (all from Task 1)
- Produces: `CalibrationResult` dataclass (`homography: np.ndarray | None`,
  `num_keypoints: int`, `num_lines: int`, `top_kp_score: float`,
  `top_line_score: float`), `PitchCalibrator` class with
  `calibrate(frame: np.ndarray) -> CalibrationResult` — consumed by Task 3's
  `CalibrationWorker`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
"""Tests for PitchCalibrator -- single-frame pixel-to-pitch homography.

Uses tests/videos/match_4.mp4 (broadcast-style footage that Phase 0 verified
calibrates reliably -- see docs/PITCH_CALIBRATION_SPEC.md). Slow (loads a
~270MB model, runs real inference), like tests/test_tracking_quality.py.
"""

from pathlib import Path

import cv2
import pytest

from scripts.calibration import PitchCalibrator

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calibration.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.calibration'`

- [ ] **Step 3: Write `scripts/calibration.py`**

```python
"""Single-frame pitch calibration: turns a video frame into a homography
from pixel coordinates to real-world pitch coordinates (meters), using
PnLCalib's pretrained points+lines model (see docs/PITCH_CALIBRATION_SPEC.md
for why this model and why points+lines, not point-only).

Model inference takes 0.3-1.8s/frame (MPS vs CPU) -- this class only does
one frame at a time; CalibrationWorker (in this same module) is what runs it
at a sustainable keyframe cadence in the background.
"""

from __future__ import annotations

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_calibration.py -v -m slow`
Expected: PASS (both tests)

- [ ] **Step 5: Manual visual sanity check**

Run a one-off script reprojecting the pitch template's line coordinates
(same list as upstream `inference.py`'s `lines_coords`) through
`result.homography` onto the `match_4` test frame, save as PNG, and confirm
by eye the lines land on the real pitch markings -- same check used in
Phase 0. This guards specifically against the Z-column-drop homography
derivation being subtly wrong (e.g. transposed, wrong sign) even though
`homography is not None` and shape checks pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibration.py tests/test_calibration.py
git commit -m "Add PitchCalibrator for single-frame pitch homography"
```

---

### Task 3: `CalibrationWorker` — background keyframe-rate calibration

**Files:**
- Modify: `scripts/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `PitchCalibrator`, `CalibrationResult` (Task 2)
- Produces: `CalibrationWorker` class — `__init__(calibrator, keyframe_interval)`,
  `.start()`, `.stop()`, `.submit(frame, frame_id)`, `.get_result() -> CalibrationResult | None`
  — consumed by Task 4's `PlayerTracker` wiring.

- [ ] **Step 1: Write the failing test**

```python
def test_calibration_worker_updates_result_after_submit():
    """Fake calibrator (no real model) so this test is fast and deterministic --
    only exercises the worker's threading/cadence logic, not PnLCalib itself."""
    import time

    from scripts.calibration import CalibrationResult, CalibrationWorker

    class FakeCalibrator:
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

    fake = FakeCalibrator()
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
    """Submitting frames faster than the keyframe interval must not calibrate
    every single one -- that's the whole point of a keyframe cadence."""
    import time

    from scripts.calibration import CalibrationResult, CalibrationWorker

    class FakeCalibrator:
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

    fake = FakeCalibrator()
    worker = CalibrationWorker(fake, keyframe_interval=1000).start()
    try:
        for frame_id in range(1, 21):
            worker.submit(frame="fake_frame", frame_id=frame_id)
            time.sleep(0.005)
        time.sleep(0.1)
        assert fake.calls <= 1
    finally:
        worker.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calibration.py -v -k CalibrationWorker`
Expected: FAIL with `ImportError: cannot import name 'CalibrationWorker'`

- [ ] **Step 3: Add `CalibrationWorker` to `scripts/calibration.py`**

```python
import threading
import time


class CalibrationWorker:
    """Runs PitchCalibrator in the background at a fixed keyframe cadence,
    never blocking the caller. Mirrors InferenceWorker's submit/get_result
    pattern (see scripts/pipeline.py) but only actually calibrates every
    `keyframe_interval`-th submitted frame_id -- calibration is too slow
    (0.3-1.8s) to run on every frame like detection does.
    """

    def __init__(self, calibrator: "PitchCalibrator", keyframe_interval: int = 30):
        self.calibrator = calibrator
        self.keyframe_interval = keyframe_interval
        self.lock = threading.Lock()
        self.pending = None  # (frame, frame_id)
        self.latest: CalibrationResult | None = None
        self.last_calibrated_frame_id = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "CalibrationWorker":
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_calibration.py -v -k CalibrationWorker`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration.py tests/test_calibration.py
git commit -m "Add CalibrationWorker for background keyframe-rate calibration"
```

---

### Task 4: Wire `CalibrationWorker` into `PlayerTracker`

**Files:**
- Modify: `scripts/pipeline.py`

**Interfaces:**
- Consumes: `scripts.calibration.PitchCalibrator`, `scripts.calibration.CalibrationWorker`
- Produces: `PlayerTracker` now starts/stops a `CalibrationWorker` alongside
  its `InferenceWorker` and prints calibration status on exit — no other
  code consumes the homography yet (that's Phase 3's `PlayerState` wiring).

- [ ] **Step 1: Add the import and worker startup in `PlayerTracker.run()`**

In `scripts/pipeline.py`, add near the other imports:
```python
from scripts.calibration import CalibrationWorker, PitchCalibrator
```

In `PlayerTracker.run()`, right after `worker = InferenceWorker(...).start()`:
```python
        calibration_worker = CalibrationWorker(PitchCalibrator()).start()
```

And in the `finally` block, alongside `worker.stop()`:
```python
            calibration_worker.stop()
```

- [ ] **Step 2: Surface the latest calibration result each frame**

In `PlayerTracker._play()`, after `result = worker.get_result()`, add:
```python
            calibration_result = calibration_worker.get_result()
```

Pass `calibration_worker` into `_play()`'s signature (alongside `worker`)
and thread `calibration_result` through so it's available for Phase 3 later
— for Phase 1, just track it locally on `self` (e.g.
`self.latest_calibration = calibration_result`) so `run()`'s summary print
can report on it; no renderer/overlay changes in this phase.

- [ ] **Step 3: Print calibration status in `run()`'s summary block**

Alongside the existing `print(worker.stats.summary())` etc. in `run()`:
```python
            if self.latest_calibration is not None:
                homography_found = self.latest_calibration.homography is not None
                print(
                    f"Calibration: homography_found={homography_found} "
                    f"keypoints={self.latest_calibration.num_keypoints} "
                    f"lines={self.latest_calibration.num_lines}"
                )
```

- [ ] **Step 4: Manual verification**

Run: `uv run python scripts/track_players.py --source data/videos/match_4.mp4 --no-window`
(check `scripts/track_players.py --help` for the actual flag names first,
since this is being run manually, not asserted in a test)

Expected: pipeline runs to completion as before, and the final summary now
includes a `Calibration: homography_found=True ...` line.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline.py
git commit -m "Wire CalibrationWorker into PlayerTracker at keyframe rate"
```

---

## Explicitly out of scope for this plan (later phases per the spec)

- Frame-to-frame homography propagation between keyframes (Phase 2)
- Low-confidence fallback / holding last-known homography with a confidence
  flag (Phase 3)
- Any consumer of the homography (`PlayerState`, heatmaps, possession) (Phase 3)
- Anything involving `match_3`-style stand-camera footage (out of scope per
  the 2026-09-12 scope decision in the spec)
