import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person"
TRACKER_CONFIG = str(Path(__file__).parent / "botsort_custom.yaml")
MODEL_NAME = "yolov8m.mlpackage"
INFERENCE_IMGSZ = 1280
DEFAULT_VIDEO = "data/videos/sample.mp4"

NUM_TEAM_CLUSTERS = 3  # 2 teams + referee/other
TEAM_FIT_AFTER_SAMPLES = 25  # jersey-color samples collected before clusters are fixed
UNCLASSIFIED_COLOR = (180, 180, 180)  # gray, shown before a track has enough samples
JERSEY_RESAMPLE_INTERVAL = 15  # worker cycles between re-observations of a settled track


def track(model: YOLO, frame):
    """Run detection + tracking on a single frame and return raw Ultralytics results."""
    return model.track(
        frame,
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker=TRACKER_CONFIG,
        device="mps",
        conf=0.15,
        imgsz=INFERENCE_IMGSZ,
        verbose=False,
    )[0]


def extract_boxes(results):
    """Convert Ultralytics results into a plain list of (x1, y1, x2, y2, track_id).

    Transfers the whole xyxy/id tensors off the GPU in one shot rather than
    indexing per-box — per-box tensor access forces a separate device sync each
    time, which measured as the dominant worker cost (~23ms/frame at ~20 boxes).
    """
    boxes_obj = results.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []
    xyxy = boxes_obj.xyxy.cpu().numpy().astype(int)
    ids = (
        boxes_obj.id.cpu().numpy().astype(int)
        if boxes_obj.id is not None
        else np.full(len(xyxy), -1, dtype=int)
    )
    return [
        (int(x1), int(y1), int(x2), int(y2), int(tid))
        for (x1, y1, x2, y2), tid in zip(xyxy, ids, strict=True)
    ]


def extract_jersey_color(frame, x1, y1, x2, y2):
    """Sample the dominant jersey color from a player's bounding box.

    Crops to the torso (avoids head/legs/background at the box edges) and masks
    out grass green so the pitch behind the player doesn't skew the color.
    """
    h = y2 - y1
    w = x2 - x1
    ty1 = max(0, y1 + int(h * 0.15))
    ty2 = max(0, y1 + int(h * 0.55))
    tx1 = max(0, x1 + int(w * 0.2))
    tx2 = max(0, x1 + int(w * 0.8))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    non_green = crop.reshape(-1, 3)[cv2.bitwise_not(green_mask).reshape(-1) > 0]
    pixels = non_green if len(non_green) >= 10 else crop.reshape(-1, 3)
    return np.median(pixels, axis=0)


def boxes_overlap(box_a, box_b, overlap_ratio_threshold=0.2, iou_threshold=0.15) -> bool:
    """True if two boxes overlap enough that jersey-color sampling would risk
    picking up the other player — e.g. two opposing players contesting a ball.
    Checked both as a fraction of the smaller box's area (catches a small box mostly
    swallowed by a bigger one) and as IoU (catches two similarly-sized boxes
    overlapping less than fully).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    if intersection == 0:
        return False

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    overlap_ratio = intersection / min(area_a, area_b)
    iou = intersection / (area_a + area_b - intersection)
    return overlap_ratio > overlap_ratio_threshold or iou > iou_threshold


class TeamClassifier:
    """Assigns each tracked player to a team based on jersey color.

    Cluster centers are fit once from the first batch of samples, then frozen —
    every later frame assigns players to the nearest *fixed* center rather than
    re-clustering, so a team's identity never flips between frames. A rolling
    average per track ID smooths out noisy single-frame color reads.
    """

    def __init__(
        self,
        k=NUM_TEAM_CLUSTERS,
        fit_after_distinct_tracks=TEAM_FIT_AFTER_SAMPLES,
        min_observations_before_fit_eligible=3,
        smoothing=0.7,
    ):
        self.k = k
        self.fit_after_distinct_tracks = fit_after_distinct_tracks
        self.min_observations_before_fit_eligible = min_observations_before_fit_eligible
        self.smoothing = smoothing
        self.centers = None
        self.track_colors = {}
        self.observation_counts = {}

    def observe(self, track_id: int, color):
        if color is None:
            return
        smoothed = self.track_colors.get(track_id)
        smoothed = (
            color if smoothed is None else self.smoothing * smoothed + (1 - self.smoothing) * color
        )
        self.track_colors[track_id] = smoothed
        self.observation_counts[track_id] = self.observation_counts.get(track_id, 0) + 1

        if self.centers is None:
            self._try_fit()

    def team_for(self, track_id: int):
        if self.centers is None:
            return None
        color = self.track_colors.get(track_id)
        if color is None:
            return None
        distances = np.linalg.norm(self.centers - color, axis=1)
        return int(np.argmin(distances))

    def team_color(self, team_id: int):
        """The team's actual average jersey color (BGR), for marker rendering."""
        b, g, r = self.centers[team_id]
        return (int(b), int(g), int(r))

    def _try_fit(self):
        """Fit once we've seen enough *distinct* tracks with a stable color estimate
        each (not just enough total observations, which a handful of repeatedly-seen
        tracks could satisfy on their own with zero diversity).
        """
        eligible_tracks = [
            track_id
            for track_id, count in self.observation_counts.items()
            if count >= self.min_observations_before_fit_eligible
        ]
        if len(eligible_tracks) < self.fit_after_distinct_tracks:
            return

        data = np.array([self.track_colors[t] for t in eligible_tracks], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, _, centers = cv2.kmeans(data, self.k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self.centers = centers


class MarkerRenderer:
    """Draws each player as a small downward-pointing pin floating a clear gap
    above the player's head (FIFA-broadcast style), rather than touching it —
    a smaller take on the original overhead triangle marker. Size scales down
    with bbox height so distant players get smaller markers.

    The anchor point (tip of the pin) is smoothed with its own, more aggressive
    exponential average than DisplaySmoother's box easing: a standing/running
    player's raw detection box wiggles a few pixels frame to frame (pose changes,
    detector noise), which DisplaySmoother's alpha=0.5 (tuned for the box to stay
    responsive, not for visual stillness) still lets most of through — enough for
    a marker floating a fixed spot above the head to visibly wiggle. A slower,
    marker-only anchor average removes that jitter without touching the box easing
    other consumers (extrapolation, coasting) rely on.
    """

    MIN_HALF_WIDTH = 8
    MAX_HALF_WIDTH = 14
    HALF_WIDTH_FROM_HEIGHT = 0.12
    PIN_HEIGHT_RATIO = 2.2  # pin height as a multiple of its half-width
    MIN_HEAD_GAP = 10
    MAX_HEAD_GAP = 20
    HEAD_GAP_FROM_HEIGHT = 0.15
    MARKER_ALPHA = 0.85
    OUTLINE_COLOR = (0, 0, 0)
    ANCHOR_EASING = 0.2
    ANCHOR_SNAP_DISTANCE_PX = 150

    def __init__(self, classifier: TeamClassifier):
        self.classifier = classifier
        self.anchor_positions = {}  # track_id -> (tip_x, tip_y) floats

    def draw(self, frame, boxes, alphas=None):
        current_ids = set()
        for x1, y1, x2, y2, track_id in boxes:
            if track_id < 0:
                continue  # unconfirmed BoT-SORT detection, not a real player ID yet
            current_ids.add(track_id)
            alpha = self.MARKER_ALPHA if alphas is None else alphas.get(track_id, self.MARKER_ALPHA)
            if alpha <= 0.02:
                continue  # fully faded out, nothing to draw
            color = self._color_for(track_id)
            tip = self._smoothed_tip(track_id, x1, y1, x2, y2)
            bbox_height = y2 - y1
            self._draw_marker(frame, tip, bbox_height, color, alpha)
            self._draw_label(frame, x1, tip, bbox_height, track_id, color)
        self._prune_anchors(current_ids)

    def _color_for(self, track_id):
        team = self.classifier.team_for(track_id)
        if team is None:
            return UNCLASSIFIED_COLOR
        return self.classifier.team_color(team)

    def _smoothed_tip(self, track_id, x1, y1, x2, y2):
        head_gap = np.clip(
            (y2 - y1) * self.HEAD_GAP_FROM_HEIGHT, self.MIN_HEAD_GAP, self.MAX_HEAD_GAP
        )
        raw_tip = ((x1 + x2) / 2, y1 - head_gap)
        previous = self.anchor_positions.get(track_id)
        if previous is None or math.hypot(raw_tip[0] - previous[0], raw_tip[1] - previous[1]) > (
            self.ANCHOR_SNAP_DISTANCE_PX
        ):
            smoothed = raw_tip
        else:
            smoothed = (
                previous[0] + (raw_tip[0] - previous[0]) * self.ANCHOR_EASING,
                previous[1] + (raw_tip[1] - previous[1]) * self.ANCHOR_EASING,
            )
        self.anchor_positions[track_id] = smoothed
        return (int(round(smoothed[0])), int(round(smoothed[1])))

    def _prune_anchors(self, current_ids):
        for track_id in [tid for tid in self.anchor_positions if tid not in current_ids]:
            del self.anchor_positions[track_id]

    def _draw_marker(self, frame, tip, bbox_height, color, alpha):
        half_width = int(
            np.clip(
                bbox_height * self.HALF_WIDTH_FROM_HEIGHT, self.MIN_HALF_WIDTH, self.MAX_HALF_WIDTH
            )
        )
        pin_height = int(half_width * self.PIN_HEIGHT_RATIO)

        points = np.array(
            [
                [tip[0], tip[1]],
                [tip[0] - half_width, tip[1] - pin_height],
                [tip[0] + half_width, tip[1] - pin_height],
            ],
            dtype=np.int32,
        )
        self._blend_triangle(frame, points, color, alpha)

    def _blend_triangle(self, frame, points, color, alpha):
        h, w = frame.shape[:2]
        pad = 2
        x0, y0 = max(0, points[:, 0].min() - pad), max(0, points[:, 1].min() - pad)
        x1, y1 = min(w, points[:, 0].max() + pad), min(h, points[:, 1].max() + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        overlay = roi.copy()
        shifted = points - [x0, y0]
        cv2.fillPoly(overlay, [shifted], color, lineType=cv2.LINE_AA)
        cv2.polylines(
            overlay,
            [shifted],
            isClosed=True,
            color=self.OUTLINE_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

    def _draw_label(self, frame, x1, tip, bbox_height, track_id, color):
        half_width = int(
            np.clip(
                bbox_height * self.HALF_WIDTH_FROM_HEIGHT, self.MIN_HALF_WIDTH, self.MAX_HALF_WIDTH
            )
        )
        pin_height = int(half_width * self.PIN_HEIGHT_RATIO)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(0, tip[1] - pin_height - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


@dataclass
class InferenceResult:
    boxes: list
    frame_id: int
    captured_at: float
    previous_boxes: list
    previous_captured_at: float | None
    coasting_progress: dict = None  # track_id -> fraction (0-1) through its grace window


class WorkerStats:
    """Breaks down where worker time actually goes: pure YOLO+tracker inference vs.
    total per-frame processing (inference + jersey color extraction + team
    classification), plus how many submitted frames the latest-frame buffer ended up
    dropping before the worker could get to them.
    """

    def __init__(self):
        self.frames_submitted = 0
        self.frames_skipped = 0
        self.inference_ms = []
        self.extract_ms = []
        self.jersey_ms = []
        self.residual_ms = []
        self.total_ms = []
        self.boxes_seen = 0
        self.jersey_extractions = 0

    def record_submit(self, was_pending_overwritten: bool):
        self.frames_submitted += 1
        if was_pending_overwritten:
            self.frames_skipped += 1

    def record_processed(
        self, inference_ms: float, extract_ms: float, jersey_ms: float, total_ms: float
    ):
        self.inference_ms.append(inference_ms)
        self.extract_ms.append(extract_ms)
        self.jersey_ms.append(jersey_ms)
        self.residual_ms.append(total_ms - inference_ms - extract_ms - jersey_ms)
        self.total_ms.append(total_ms)

    def summary(self) -> str:
        frames_processed = len(self.total_ms)
        lines = [
            f"Frames submitted: {self.frames_submitted}",
            f"Frames skipped (overwritten before processing): {self.frames_skipped}",
            f"Frames processed by worker: {frames_processed}",
        ]
        if frames_processed:

            def stat(name, values):
                arr = np.array(values)
                return f"{name}: avg={arr.mean():.1f} min={arr.min():.1f} max={arr.max():.1f}"

            inf = np.array(self.inference_ms)
            lines.append(
                f"YOLO+tracker inference latency (ms): avg={inf.mean():.1f} "
                f"min={inf.min():.1f} max={inf.max():.1f}"
            )
            lines.append(f"Effective inference FPS: {1000 / inf.mean():.1f}")
            lines.append(stat("extract_boxes (ms)", self.extract_ms))
            lines.append(stat("jersey extraction loop (ms)", self.jersey_ms))
            lines.append(stat("residual/unaccounted (ms)", self.residual_ms))
            lines.append(stat("Worker total latency (ms)", self.total_ms))
            lines.append(
                f"Jersey extractions: {self.jersey_extractions}/{self.boxes_seen} boxes seen "
                f"({100 * self.jersey_extractions / max(1, self.boxes_seen):.1f}%)"
            )
        return "\n".join(lines)


class StateManager:
    """Confirms a track as visible only after it's been detected for CONFIRM_CYCLES
    consecutive worker cycles, so a single spurious/noise detection doesn't blink
    onto screen for one frame and vanish. Operates on worker cycles (AI results),
    not display frames, since a single result is redisplayed across several display
    frames via extrapolation.

    Once a track is confirmed, a lone missed detection (motion blur, brief occlusion,
    ordinary detector noise) coasts on its last known box for up to GRACE_CYCLES
    cycles instead of vanishing immediately and having to re-earn CONFIRM_CYCLES from
    scratch — measured on real footage, an unconditional instant-drop caused a
    confirmed track dropout roughly once per worker cycle, i.e. constant flicker.
    """

    CONFIRM_CYCLES = 2
    GRACE_CYCLES = 2

    def __init__(self):
        self.consecutive_counts = {}  # track_id -> consecutive cycles detected
        self.miss_counts = {}  # track_id -> consecutive cycles missed since last seen
        self.last_box = {}  # track_id -> last known (x1, y1, x2, y2), while confirmed

    def update(self, boxes):
        """Call once per worker cycle with that cycle's raw detected boxes. Returns
        the subset of boxes for tracks that have reached CONFIRM_CYCLES, including
        confirmed tracks currently coasting through a within-grace miss."""
        present_ids = set()
        confirmed = []
        for x1, y1, x2, y2, track_id in boxes:
            if track_id < 0:
                confirmed.append((x1, y1, x2, y2, track_id))
                continue
            present_ids.add(track_id)
            count = self.consecutive_counts.get(track_id, 0) + 1
            self.consecutive_counts[track_id] = count
            self.miss_counts[track_id] = 0
            if count >= self.CONFIRM_CYCLES:
                self.last_box[track_id] = (x1, y1, x2, y2)
                confirmed.append((x1, y1, x2, y2, track_id))

        for track_id in list(self.consecutive_counts):
            if track_id in present_ids:
                continue
            if track_id not in self.last_box:
                del self.consecutive_counts[track_id]  # never confirmed, no grace
                continue
            self.miss_counts[track_id] += 1
            if self.miss_counts[track_id] > self.GRACE_CYCLES:
                del self.consecutive_counts[track_id]
                del self.miss_counts[track_id]
                del self.last_box[track_id]
            else:
                x1, y1, x2, y2 = self.last_box[track_id]
                confirmed.append((x1, y1, x2, y2, track_id))

        return confirmed


class InferenceWorker:
    """Runs detection + tracking continuously in the background, always working on
    the most recently submitted frame. Frames arriving faster than inference finishes
    are dropped (never queued), so the worker never falls further and further behind.

    Each submitted frame carries a frame_id and capture timestamp, which is echoed
    back with the result — this lets the caller measure exactly how many frames (and
    how many milliseconds) old the boxes it's currently displaying are, and also
    extrapolate player motion forward using the two most recent results (see
    MotionExtrapolator).
    """

    def __init__(self, model: YOLO, classifier: TeamClassifier):
        self.model = model
        self.classifier = classifier
        self.lock = threading.Lock()
        self.pending = None  # (frame, frame_id, captured_at)
        self.latest = InferenceResult(
            boxes=[],
            frame_id=None,
            captured_at=None,
            previous_boxes=[],
            previous_captured_at=None,
            coasting_progress={},
        )
        self.running = True
        self.stats = WorkerStats()
        self.state = StateManager()
        self.cycle_count = 0
        self.jersey_last_sampled = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def submit(self, frame, frame_id: int):
        with self.lock:
            self.stats.record_submit(was_pending_overwritten=self.pending is not None)
            self.pending = (frame, frame_id, time.perf_counter())

    def get_result(self) -> InferenceResult:
        with self.lock:
            return self.latest

    def _run(self):
        while self.running:
            pending = self._take_pending()
            if pending is None:
                time.sleep(0.001)
                continue
            frame, frame_id, captured_at = pending
            total_start = time.perf_counter()

            inference_start = time.perf_counter()
            results = track(self.model, frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000

            extract_start = time.perf_counter()
            boxes = extract_boxes(results)
            extract_ms = (time.perf_counter() - extract_start) * 1000
            self.cycle_count += 1

            jersey_start = time.perf_counter()
            for i, (x1, y1, x2, y2, track_id) in enumerate(boxes):
                self.stats.boxes_seen += 1
                if track_id < 0:
                    continue
                if not self._should_sample_jersey(track_id):
                    continue
                if self._overlaps_another(boxes, i):
                    continue
                self.stats.jersey_extractions += 1
                color = extract_jersey_color(frame, x1, y1, x2, y2)
                self.classifier.observe(track_id, color)
            jersey_ms = (time.perf_counter() - jersey_start) * 1000

            total_ms = (time.perf_counter() - total_start) * 1000
            self.stats.record_processed(inference_ms, extract_ms, jersey_ms, total_ms)

            confirmed_boxes = self.state.update(boxes)
            coasting_progress = {
                track_id: min(1.0, miss_count / self.state.GRACE_CYCLES)
                for track_id, miss_count in self.state.miss_counts.items()
                if miss_count > 0
            }

            with self.lock:
                self.latest = InferenceResult(
                    boxes=confirmed_boxes,
                    frame_id=frame_id,
                    captured_at=captured_at,
                    previous_boxes=self.latest.boxes,
                    previous_captured_at=self.latest.captured_at,
                    coasting_progress=coasting_progress,
                )

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending

    def _overlaps_another(self, boxes, index) -> bool:
        x1, y1, x2, y2, _ = boxes[index]
        return any(
            boxes_overlap((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
            for i, (ox1, oy1, ox2, oy2, _) in enumerate(boxes)
            if i != index
        )

    def _should_sample_jersey(self, track_id) -> bool:
        # Always sample while the track is still unclassified or clusters aren't fit yet
        if self.classifier.team_for(track_id) is None:
            return True
        # Settled track: re-sample only every Nth worker cycle
        last = self.jersey_last_sampled.get(track_id, -JERSEY_RESAMPLE_INTERVAL)
        if self.cycle_count - last >= JERSEY_RESAMPLE_INTERVAL:
            self.jersey_last_sampled[track_id] = self.cycle_count
            return True
        return False


class MotionExtrapolator:
    """Shifts each player's last known box forward in time using the velocity
    estimated between the two most recent AI results, so the displayed marker keeps
    moving smoothly between inference updates instead of freezing at a stale position.

    A max shift cap guards against runaway extrapolation from a noisy velocity
    estimate (e.g. an ID that just switched, or a very short dt between results).
    """

    def __init__(self, max_shift_px=120):
        self.max_shift_px = max_shift_px

    def extrapolate(self, result: InferenceResult, now: float):
        if result.captured_at is None or result.previous_captured_at is None:
            return result.boxes

        dt = result.captured_at - result.previous_captured_at
        elapsed = now - result.captured_at
        if dt <= 0 or elapsed <= 0:
            return result.boxes

        previous_by_id = {
            track_id: (x1, y1, x2, y2) for x1, y1, x2, y2, track_id in result.previous_boxes
        }

        extrapolated = []
        for x1, y1, x2, y2, track_id in result.boxes:
            previous = previous_by_id.get(track_id)
            if previous is None:
                extrapolated.append((x1, y1, x2, y2, track_id))
                continue

            px1, py1, _, _ = previous
            shift_x = self._clamp((x1 - px1) / dt * elapsed)
            shift_y = self._clamp((y1 - py1) / dt * elapsed)
            extrapolated.append((x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y, track_id))
        return extrapolated

    def _clamp(self, shift: float) -> int:
        return int(max(-self.max_shift_px, min(self.max_shift_px, shift)))


class DisplaySmoother:
    """Eases each track's displayed box position toward its latest target instead of
    jumping straight to it, so a fresh AI/extrapolation result doesn't read as a
    micro-teleport. Position is floated end-to-end and only rounded at draw time so
    easing doesn't stall on integer rounding.

    Snaps instead of gliding for a track's first sighting (nothing to ease from) or
    when the target jumps further than a real player could move in one cycle (an ID
    switch reusing a track_id at a new location) — gliding across an ID switch would
    look like the marker sliding across the pitch.
    """

    EASING = 0.5
    SNAP_DISTANCE_PX = 150

    def __init__(self):
        self.positions = {}  # track_id -> (x1, y1, x2, y2) floats

    def smooth(self, boxes):
        current_ids = set()
        smoothed = []
        for x1, y1, x2, y2, track_id in boxes:
            if track_id < 0:
                smoothed.append((x1, y1, x2, y2, track_id))
                continue

            current_ids.add(track_id)
            target = (float(x1), float(y1), float(x2), float(y2))
            previous = self.positions.get(track_id)
            if previous is None or self._jumped(previous, target):
                eased = target
            else:
                eased = tuple(
                    p + (t - p) * self.EASING for p, t in zip(previous, target, strict=True)
                )
            self.positions[track_id] = eased
            smoothed.append((*(int(round(v)) for v in eased), track_id))

        self._prune(current_ids)
        return smoothed

    def _jumped(self, previous, target) -> bool:
        px1, py1, px2, py2 = previous
        tx1, ty1, tx2, ty2 = target
        dist = math.hypot((tx1 + tx2) / 2 - (px1 + px2) / 2, (ty1 + ty2) / 2 - (py1 + py2) / 2)
        return dist > self.SNAP_DISTANCE_PX

    def _prune(self, current_ids):
        for track_id in [tid for tid in self.positions if tid not in current_ids]:
            del self.positions[track_id]


class FadeController:
    """Ramps each track's marker opacity in/out across display frames instead of
    cutting it in or out instantly, so a newly confirmed track fades in and a track
    coasting through the back half of its StateManager grace window fades out —
    appearances/disappearances read as fades, not cuts.

    Operates per display frame (like DisplaySmoother), driven by the latest worker
    result's coasting_progress: a track past half its grace window (progress > 0.5)
    ramps toward 0; every other visible track ramps toward TARGET_ALPHA.
    """

    TARGET_ALPHA = MarkerRenderer.MARKER_ALPHA
    FADE_IN_FRAMES = 5
    FADE_OUT_FRAMES = 5

    def __init__(self):
        self.opacity = {}  # track_id -> current alpha

    def update(self, boxes, coasting_progress):
        coasting_progress = coasting_progress or {}
        current_ids = set()
        alphas = {}
        for *_, track_id in boxes:
            if track_id < 0:
                continue
            current_ids.add(track_id)
            current = self.opacity.get(track_id, 0.0)
            fading_out = coasting_progress.get(track_id, 0.0) > 0.5
            target = 0.0 if fading_out else self.TARGET_ALPHA
            frames = self.FADE_OUT_FRAMES if fading_out else self.FADE_IN_FRAMES
            step = self.TARGET_ALPHA / frames
            if current < target:
                current = min(target, current + step)
            elif current > target:
                current = max(target, current - step)
            self.opacity[track_id] = current
            alphas[track_id] = current

        for track_id in [tid for tid in self.opacity if tid not in current_ids]:
            del self.opacity[track_id]
        return alphas


class StalenessTracker:
    """Measures how far behind the AI result being displayed is from the current
    frame — in both frame count and wall-clock time. This quantifies the "markers
    lag the real player position" effect inherent to async inference.
    """

    def __init__(self):
        self.frame_gaps = []
        self.ms_gaps = []

    def record(self, current_frame_id: int, result_frame_id, result_captured_at):
        if result_frame_id is None:
            return  # no AI result yet
        self.frame_gaps.append(current_frame_id - result_frame_id)
        self.ms_gaps.append((time.perf_counter() - result_captured_at) * 1000)

    def summary(self) -> str:
        if not self.frame_gaps:
            return "Staleness: no AI results were ever displayed"
        return (
            "Staleness (frames behind): "
            f"avg={sum(self.frame_gaps) / len(self.frame_gaps):.1f} "
            f"min={min(self.frame_gaps)} max={max(self.frame_gaps)}\n"
            "Staleness (ms behind): "
            f"avg={sum(self.ms_gaps) / len(self.ms_gaps):.1f}ms "
            f"min={min(self.ms_gaps):.1f}ms max={max(self.ms_gaps):.1f}ms"
        )


class DisplayStats:
    """Times each segment of the main display loop (frame read, submitting to the
    worker, drawing markers, cv2.imshow, and the pacer's wait) so we can see exactly
    where main-thread time goes, rather than assuming it's all in one place.
    """

    def __init__(self):
        self.read_ms = []
        self.submit_ms = []
        self.draw_ms = []
        self.imshow_ms = []
        self.wait_ms = []
        self.pre_wait_overrun_count = 0
        self.frame_budget_ms = None

    def record(self, read_ms, submit_ms, draw_ms, imshow_ms, wait_ms, frame_budget_ms):
        self.frame_budget_ms = frame_budget_ms
        self.read_ms.append(read_ms)
        self.submit_ms.append(submit_ms)
        self.draw_ms.append(draw_ms)
        self.imshow_ms.append(imshow_ms)
        self.wait_ms.append(wait_ms)
        pre_wait = read_ms + submit_ms + draw_ms + imshow_ms
        if pre_wait > frame_budget_ms:
            self.pre_wait_overrun_count += 1

    def summary(self) -> str:
        if not self.read_ms:
            return "Display timing: no frames were ever displayed"

        def stat(name, values):
            arr = np.array(values)
            return f"{name}: avg={arr.mean():.1f} max={arr.max():.1f}"

        lines = [
            stat("read (ms)", self.read_ms),
            stat("submit (ms)", self.submit_ms),
            stat("draw (extrapolate+render) (ms)", self.draw_ms),
            stat("imshow (ms)", self.imshow_ms),
            stat("wait (ms)", self.wait_ms),
            f"Pre-wait work exceeding {self.frame_budget_ms:.1f}ms budget: "
            f"{self.pre_wait_overrun_count}/{len(self.read_ms)} frames",
        ]
        return "\n".join(lines)


class FpsOverlay:
    """Tracks a smoothed live display FPS and draws it on the frame (top-left)
    against the video's own native FPS, so playback speed is visible during
    playback rather than only printed to the terminal after the run ends.

    Colored green while live FPS tracks native FPS, and red once it falls below
    DROP_RATIO of native — the same threshold the pytest FPS regression test
    uses — so a real slowdown is visually obvious rather than requiring the
    viewer to compare two numbers themselves.

    The underlying value is smoothed and updated every frame for accuracy, but
    the *displayed* text only refreshes every REFRESH_INTERVAL_S — redrawing a
    changed digit every single frame (~24-60x/sec) reads as flicker even when
    the smoothed value itself is barely moving.
    """

    SMOOTHING = 0.9  # closer to 1 = smoother/slower-reacting, less noisy readout
    REFRESH_INTERVAL_S = 0.5  # how often the on-screen text is allowed to change
    DROP_RATIO = 0.8
    POSITION = (10, 30)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    COLOR_OK = (0, 255, 0)
    COLOR_DROPPED = (0, 0, 255)
    THICKNESS = 2

    def __init__(self):
        self.smoothed_fps = None
        self.displayed_fps = None
        self.last_tick_at = None
        self.last_refresh_at = None

    def tick(self, now: float | None = None) -> float | None:
        now = time.perf_counter() if now is None else now
        if self.last_tick_at is not None:
            dt = now - self.last_tick_at
            if dt > 0:
                instant_fps = 1.0 / dt
                self.smoothed_fps = (
                    instant_fps
                    if self.smoothed_fps is None
                    else self.SMOOTHING * self.smoothed_fps + (1 - self.SMOOTHING) * instant_fps
                )
        self.last_tick_at = now

        if self.displayed_fps is None or now - self.last_refresh_at >= self.REFRESH_INTERVAL_S:
            self.displayed_fps = self.smoothed_fps
            self.last_refresh_at = now
        return self.smoothed_fps

    def draw(self, frame, native_fps: float):
        if self.displayed_fps is None:
            return
        dropped = self.displayed_fps < self.DROP_RATIO * native_fps
        color = self.COLOR_DROPPED if dropped else self.COLOR_OK
        text = f"Live FPS: {self.displayed_fps:.1f}  |  Video FPS: {native_fps:.1f}"
        cv2.putText(
            frame,
            text,
            self.POSITION,
            self.FONT,
            self.FONT_SCALE,
            color,
            self.THICKNESS,
            lineType=cv2.LINE_AA,
        )


class FramePacer:
    """Paces against an absolute frame schedule (not a per-iteration budget), so
    a single frame's timing error self-corrects on the next frame instead of
    compounding. Per-iteration budgeting drifted in practice: cv2.waitKey()
    overshoots its requested delay by ~4-5ms on macOS (it also pumps the GUI
    event loop), and since that overshoot was never repaid, avg wait crept up
    to ~43ms against a 41.7ms budget — a small, silent, permanent FPS loss.

    Re-anchors the schedule if playback falls behind by more than a few frames
    (e.g. after a long stall), rather than trying to burn through a backlog of
    missed deadlines in a rapid-fire burst.
    """

    REANCHOR_AFTER_FRAMES_BEHIND = 3
    WAITKEY_OVERSHOOT_MS = 6  # observed average cv2.waitKey overshoot on macOS

    def __init__(self, fps: float):
        self.frame_budget_ms = 1000 / fps
        self.frame_interval = 1 / fps
        self.next_deadline = None

    def wait(self, _iteration_start: float, use_gui: bool = True) -> int:
        now = time.perf_counter()
        if self.next_deadline is None:
            self.next_deadline = now + self.frame_interval
        elif now - self.next_deadline > self.REANCHOR_AFTER_FRAMES_BEHIND * self.frame_interval:
            self.next_deadline = now + self.frame_interval

        remaining_ms = (self.next_deadline - now) * 1000
        self.next_deadline += self.frame_interval

        wait_ms = max(1, int(remaining_ms - self.WAITKEY_OVERSHOOT_MS))
        if use_gui:
            return cv2.waitKey(wait_ms) & 0xFF
        time.sleep(wait_ms / 1000)
        return -1


class PlayerTracker:
    """Plays a video live, overlaying player tracking markers.

    Detection/tracking runs on a background thread (see InferenceWorker) so
    display always paces at the video's real frame rate, independent of how
    long a single inference call takes.
    """

    WINDOW_NAME = "Football Tracker"
    QUIT_KEY = ord("q")

    def __init__(self, video_path: Path, model: YOLO, show_window: bool = True):
        self.video_path = video_path
        self.model = model
        self.show_window = show_window
        self.classifier = TeamClassifier()
        self.renderer = MarkerRenderer(self.classifier)
        self.extrapolator = MotionExtrapolator()
        self.smoother = DisplaySmoother()
        self.fader = FadeController()
        self.staleness = StalenessTracker()
        self.display_stats = DisplayStats()
        self.fps_overlay = FpsOverlay()
        self.frame_count = 0
        self.play_start = None
        self.achieved_fps = None

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self._warmup()

        worker = InferenceWorker(self.model, self.classifier).start()
        self.play_start = time.perf_counter()
        try:
            self._play(cap, worker, pacer)
        finally:
            worker.stop()
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            elapsed = time.perf_counter() - self.play_start
            self.achieved_fps = self.frame_count / elapsed
            print(f"Read {self.frame_count} frames from {self.video_path}")
            print(f"Display FPS: {self.achieved_fps:.1f}")
            print()
            print(worker.stats.summary())
            print()
            print(self.staleness.summary())
            print()
            print(self.display_stats.summary())

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            sys.exit(1)
        return cap

    def _warmup(self):
        """Pay the one-time GPU kernel compilation cost (MPS) before playback starts."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track(self.model, blank_frame)

    def _play(self, cap: cv2.VideoCapture, worker: InferenceWorker, pacer: FramePacer):
        while True:
            iteration_start = time.perf_counter()
            self.fps_overlay.tick(iteration_start)

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            t1 = time.perf_counter()

            worker.submit(frame.copy(), self.frame_count)
            result = worker.get_result()
            t2 = time.perf_counter()
            now = time.perf_counter()
            self.staleness.record(self.frame_count, result.frame_id, result.captured_at)

            boxes = self.extrapolator.extrapolate(result, now)
            boxes = self.smoother.smooth(boxes)
            alphas = self.fader.update(boxes, result.coasting_progress)
            self.renderer.draw(frame, boxes, alphas)
            t3 = time.perf_counter()
            if self.show_window:
                self.fps_overlay.draw(frame, native_fps=1000 / pacer.frame_budget_ms)
                cv2.imshow(self.WINDOW_NAME, frame)
            t4 = time.perf_counter()

            key = pacer.wait(iteration_start, use_gui=self.show_window)
            t5 = time.perf_counter()

            self.display_stats.record(
                read_ms=(t1 - t0) * 1000,
                submit_ms=(t2 - t1) * 1000,
                draw_ms=(t3 - t2) * 1000,
                imshow_ms=(t4 - t3) * 1000,
                wait_ms=(t5 - t4) * 1000,
                frame_budget_ms=pacer.frame_budget_ms,
            )

            if self.show_window and key == self.QUIT_KEY:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Track players in a football video")
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Path to the input video (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument("--model", default=MODEL_NAME, help=f"YOLO model (default: {MODEL_NAME})")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=INFERENCE_IMGSZ,
        help=f"Inference size (default: {INFERENCE_IMGSZ})",
    )
    return parser.parse_args()


def main():
    global MODEL_NAME, INFERENCE_IMGSZ
    args = parse_args()
    MODEL_NAME = args.model
    INFERENCE_IMGSZ = args.imgsz

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Place a fixed-camera football video there, or pass a path as an argument.")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    PlayerTracker(video_path, model).run()


if __name__ == "__main__":
    main()
