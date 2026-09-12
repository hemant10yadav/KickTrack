import argparse
import dataclasses
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
COASTING_GRACE_SECONDS = 0.75  # how long a missed player still renders (extrapolated)
STATE_EVICT_AFTER_SECONDS = (
    5.0  # prune tracks not seen this long (bounds memory, avoids stale-ID reuse)
)


def track(model: YOLO, frame):
    """Run detection + tracking on a single frame and return raw Ultralytics results."""
    return model.track(
        frame,
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker=TRACKER_CONFIG,
        device="mps",  # ignored for the CoreML backend (.mlpackage picks its own compute unit)
        conf=0.15,
        imgsz=INFERENCE_IMGSZ,
        verbose=False,
    )[0]


def extract_boxes(results):
    """Convert Ultralytics results into a plain list of
    (x1, y1, x2, y2, track_id, confidence).

    Transfers the whole xyxy/id/conf tensors off the GPU in one shot rather than
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
    confs = boxes_obj.conf.cpu().numpy()
    return [
        (int(x1), int(y1), int(x2), int(y2), int(tid), float(conf))
        for (x1, y1, x2, y2), tid, conf in zip(xyxy, ids, confs, strict=True)
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


@dataclass
class PlayerState:
    """Canonical per-player data. Everything downstream (rendering, extrapolation,
    and eventually match analytics — distance, speed, heatmaps) reads from this
    instead of passing around raw (x1, y1, x2, y2, track_id) tuples.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    position: tuple[float, float]  # bottom-center of bbox (the feet), in pixels
    velocity: tuple[float, float]  # pixels/sec, smoothed
    team: int | None
    confidence: float
    last_seen_frame: int
    last_seen_at: float  # time.perf_counter() timestamp
    is_coasting: bool = False  # True if not detected this cycle, riding the grace window


class StateManager:
    """Maintains the canonical PlayerState per track, updated from each worker
    cycle's detections. Computes smoothed velocity from consecutive position
    updates for the same track — this replaces the old two-result-diff velocity
    calc that used to live in MotionExtrapolator, and is now the single source of
    truth for position/velocity/team that everything else reads from.
    """

    VELOCITY_SMOOTHING = 0.7

    def __init__(self, classifier: TeamClassifier):
        self.classifier = classifier
        self.states: dict[int, PlayerState] = {}

    def update(self, detections, frame_id: int, captured_at: float) -> dict[int, PlayerState]:
        """detections: list of (x1, y1, x2, y2, track_id, confidence).

        Updates state for every track detected this cycle, then returns all
        *visible* states — including players missed this cycle but seen recently
        enough to still coast on their last known position/velocity (see
        COASTING_GRACE_SECONDS). A player YOLO misses for a single cycle shouldn't
        instantly vanish; that grace window is what prevents it.
        """
        current_ids = set()
        for x1, y1, x2, y2, track_id, confidence in detections:
            if track_id < 0:
                continue
            current_ids.add(track_id)
            position = ((x1 + x2) / 2, float(y2))
            velocity = self._compute_velocity(track_id, position, captured_at)
            self.states[track_id] = PlayerState(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                position=position,
                velocity=velocity,
                team=self.classifier.team_for(track_id),
                confidence=confidence,
                last_seen_frame=frame_id,
                last_seen_at=captured_at,
                is_coasting=False,
            )

        self._evict(captured_at)
        return self.visible(captured_at, current_ids)

    def visible(
        self, now: float, current_ids=frozenset(), grace_seconds=COASTING_GRACE_SECONDS
    ) -> dict[int, PlayerState]:
        result = {}
        for track_id, state in self.states.items():
            if now - state.last_seen_at > grace_seconds:
                continue
            result[track_id] = (
                state if track_id in current_ids else dataclasses.replace(state, is_coasting=True)
            )
        return result

    def _evict(self, now: float, max_age=STATE_EVICT_AFTER_SECONDS):
        stale_ids = [
            track_id
            for track_id, state in self.states.items()
            if now - state.last_seen_at > max_age
        ]
        for track_id in stale_ids:
            del self.states[track_id]

    def _compute_velocity(self, track_id, position, captured_at) -> tuple[float, float]:
        previous = self.states.get(track_id)
        if previous is None:
            return (0.0, 0.0)
        dt = captured_at - previous.last_seen_at
        if dt <= 0:
            return previous.velocity
        raw_vx = (position[0] - previous.position[0]) / dt
        raw_vy = (position[1] - previous.position[1]) / dt
        old_vx, old_vy = previous.velocity
        s = self.VELOCITY_SMOOTHING
        return (s * old_vx + (1 - s) * raw_vx, s * old_vy + (1 - s) * raw_vy)


class MarkerRenderer:
    """Draws player markers (triangle + ID label), color-coded by team."""

    def __init__(self, classifier: TeamClassifier, size=14):
        self.classifier = classifier
        self.size = size

    COASTING_DIM_FACTOR = 0.5  # darken a coasting (undetected-this-cycle) marker's color

    def draw(self, frame, boxes):
        for x1, y1, x2, _y2, track_id, is_coasting in boxes:
            if track_id < 0:
                continue  # unconfirmed BoT-SORT detection, not a real player ID yet
            color = self._color_for(track_id, is_coasting)
            self._draw_triangle(frame, x1, y1, x2, color)
            self._draw_label(frame, x1, y1, track_id, color)

    def _color_for(self, track_id, is_coasting):
        team = self.classifier.team_for(track_id)
        color = UNCLASSIFIED_COLOR if team is None else self.classifier.team_color(team)
        if is_coasting:
            color = tuple(int(c * self.COASTING_DIM_FACTOR) for c in color)
        return color

    def _draw_triangle(self, frame, x1, y1, x2, color):
        center_x = (x1 + x2) // 2
        top_y = y1 - 4
        points = np.array(
            [
                [center_x, top_y],
                [center_x - self.size, top_y - self.size * 2],
                [center_x + self.size, top_y - self.size * 2],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [points], color)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 0), thickness=1)

    def _draw_label(self, frame, x1, y1, track_id, color):
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


@dataclass
class InferenceResult:
    states: dict[int, PlayerState]
    frame_id: int
    captured_at: float


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


class InferenceWorker:
    """Runs detection + tracking continuously in the background, always working on
    the most recently submitted frame. Frames arriving faster than inference finishes
    are dropped (never queued), so the worker never falls further and further behind.

    Each submitted frame carries a frame_id and capture timestamp, which is echoed
    back with the result — this lets the caller measure exactly how many frames (and
    how many milliseconds) old the state it's currently displaying is, and also
    extrapolate player motion forward using each PlayerState's own velocity (see
    MotionExtrapolator).
    """

    def __init__(self, model: YOLO, classifier: TeamClassifier, state_manager: StateManager):
        self.model = model
        self.classifier = classifier
        self.state_manager = state_manager
        self.lock = threading.Lock()
        self.pending = None  # (frame, frame_id, captured_at)
        self.latest = InferenceResult(states={}, frame_id=None, captured_at=None)
        self.running = True
        self.stats = WorkerStats()
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
            for x1, y1, x2, y2, track_id, _conf in boxes:
                self.stats.boxes_seen += 1
                if track_id < 0:
                    continue
                if not self._should_sample_jersey(track_id):
                    continue
                self.stats.jersey_extractions += 1
                color = extract_jersey_color(frame, x1, y1, x2, y2)
                self.classifier.observe(track_id, color)
            jersey_ms = (time.perf_counter() - jersey_start) * 1000

            states = self.state_manager.update(boxes, frame_id, captured_at)

            total_ms = (time.perf_counter() - total_start) * 1000
            self.stats.record_processed(inference_ms, extract_ms, jersey_ms, total_ms)

            with self.lock:
                self.latest = InferenceResult(
                    states=states,
                    frame_id=frame_id,
                    captured_at=captured_at,
                )

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending

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
    """Shifts each player's last known box forward in time using that PlayerState's
    own smoothed velocity, so the displayed marker keeps moving between inference
    updates instead of freezing at a stale position.

    A max shift cap guards against runaway extrapolation from a noisy velocity
    estimate (e.g. an ID that just switched, or a very short dt between updates).
    """

    def __init__(self, max_shift_px=120):
        self.max_shift_px = max_shift_px

    def extrapolate(self, states: dict[int, PlayerState], now: float):
        extrapolated = []
        for state in states.values():
            x1, y1, x2, y2 = state.bbox
            elapsed = now - state.last_seen_at
            if elapsed <= 0:
                extrapolated.append((x1, y1, x2, y2, state.track_id, state.is_coasting))
                continue

            vx, vy = state.velocity
            shift_x = self._clamp(vx * elapsed)
            shift_y = self._clamp(vy * elapsed)
            extrapolated.append(
                (
                    x1 + shift_x,
                    y1 + shift_y,
                    x2 + shift_x,
                    y2 + shift_y,
                    state.track_id,
                    state.is_coasting,
                )
            )
        return extrapolated

    def _clamp(self, shift: float) -> int:
        return int(max(-self.max_shift_px, min(self.max_shift_px, shift)))


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

    def wait(self, _iteration_start: float) -> int:
        now = time.perf_counter()
        if self.next_deadline is None:
            self.next_deadline = now + self.frame_interval
        elif now - self.next_deadline > self.REANCHOR_AFTER_FRAMES_BEHIND * self.frame_interval:
            self.next_deadline = now + self.frame_interval

        remaining_ms = (self.next_deadline - now) * 1000
        self.next_deadline += self.frame_interval

        wait_ms = max(1, int(remaining_ms - self.WAITKEY_OVERSHOOT_MS))
        return cv2.waitKey(wait_ms) & 0xFF


class PlayerTracker:
    """Plays a video live, overlaying player tracking markers.

    Detection/tracking runs on a background thread (see InferenceWorker) so
    display always paces at the video's real frame rate, independent of how
    long a single inference call takes.
    """

    WINDOW_NAME = "Football Tracker"
    QUIT_KEY = ord("q")

    def __init__(self, video_path: Path, model: YOLO):
        self.video_path = video_path
        self.model = model
        self.classifier = TeamClassifier()
        self.state_manager = StateManager(self.classifier)
        self.renderer = MarkerRenderer(self.classifier)
        self.extrapolator = MotionExtrapolator()
        self.staleness = StalenessTracker()
        self.display_stats = DisplayStats()
        self.frame_count = 0
        self.play_start = None

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self._warmup()

        worker = InferenceWorker(self.model, self.classifier, self.state_manager).start()
        self.play_start = time.perf_counter()
        try:
            self._play(cap, worker, pacer)
        finally:
            worker.stop()
            cap.release()
            cv2.destroyAllWindows()
            elapsed = time.perf_counter() - self.play_start
            print(f"Read {self.frame_count} frames from {self.video_path}")
            print(f"Display FPS: {self.frame_count / elapsed:.1f}")
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
        """Pay the one-time model compilation cost (CoreML/Neural Engine) before playback starts."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track(self.model, blank_frame)

    def _play(self, cap: cv2.VideoCapture, worker: InferenceWorker, pacer: FramePacer):
        while True:
            iteration_start = time.perf_counter()

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

            boxes = self.extrapolator.extrapolate(result.states, now)
            self.renderer.draw(frame, boxes)
            t3 = time.perf_counter()
            cv2.imshow(self.WINDOW_NAME, frame)
            t4 = time.perf_counter()

            key = pacer.wait(iteration_start)
            t5 = time.perf_counter()

            self.display_stats.record(
                read_ms=(t1 - t0) * 1000,
                submit_ms=(t2 - t1) * 1000,
                draw_ms=(t3 - t2) * 1000,
                imshow_ms=(t4 - t3) * 1000,
                wait_ms=(t5 - t4) * 1000,
                frame_budget_ms=pacer.frame_budget_ms,
            )

            if key == self.QUIT_KEY:
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
