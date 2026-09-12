import argparse
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
MODEL_NAME = "yolov8m.pt"
INFERENCE_IMGSZ = 1280
DEFAULT_VIDEO = "data/videos/sample.mp4"

NUM_TEAM_CLUSTERS = 3  # 2 teams + referee/other
TEAM_FIT_AFTER_SAMPLES = 25  # jersey-color samples collected before clusters are fixed
UNCLASSIFIED_COLOR = (180, 180, 180)  # gray, shown before a track has enough samples


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
    """Convert Ultralytics results into a plain list of (x1, y1, x2, y2, track_id)."""
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        boxes.append((x1, y1, x2, y2, track_id))
    return boxes


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


class MarkerRenderer:
    """Draws player markers (triangle + ID label), color-coded by team."""

    def __init__(self, classifier: TeamClassifier, size=14):
        self.classifier = classifier
        self.size = size

    def draw(self, frame, boxes):
        for x1, y1, x2, _y2, track_id in boxes:
            color = self._color_for(track_id)
            self._draw_triangle(frame, x1, y1, x2, color)
            self._draw_label(frame, x1, y1, track_id, color)

    def _color_for(self, track_id):
        team = self.classifier.team_for(track_id)
        if team is None:
            return UNCLASSIFIED_COLOR
        return self.classifier.team_color(team)

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
    boxes: list
    frame_id: int
    captured_at: float
    previous_boxes: list
    previous_captured_at: float | None


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
            boxes=[], frame_id=None, captured_at=None, previous_boxes=[], previous_captured_at=None
        )
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def submit(self, frame, frame_id: int):
        with self.lock:
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

            results = track(self.model, frame)
            boxes = extract_boxes(results)

            for x1, y1, x2, y2, track_id in boxes:
                color = extract_jersey_color(frame, x1, y1, x2, y2)
                self.classifier.observe(track_id, color)

            with self.lock:
                self.latest = InferenceResult(
                    boxes=boxes,
                    frame_id=frame_id,
                    captured_at=captured_at,
                    previous_boxes=self.latest.boxes,
                    previous_captured_at=self.latest.captured_at,
                )

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending


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


class FramePacer:
    """Waits out the remainder of a frame's time budget, accounting for time
    already spent this iteration, so processing cost doesn't compound into extra lag.
    """

    def __init__(self, fps: float):
        self.frame_budget_ms = 1000 / fps

    def wait(self, iteration_start: float) -> int:
        elapsed_ms = (time.perf_counter() - iteration_start) * 1000
        remaining_ms = max(1, int(self.frame_budget_ms - elapsed_ms))
        return cv2.waitKey(remaining_ms) & 0xFF


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
        self.renderer = MarkerRenderer(self.classifier)
        self.extrapolator = MotionExtrapolator()
        self.staleness = StalenessTracker()
        self.frame_count = 0

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self._warmup()

        worker = InferenceWorker(self.model, self.classifier).start()
        try:
            self._play(cap, worker, pacer)
        finally:
            worker.stop()
            cap.release()
            cv2.destroyAllWindows()
            print(f"Read {self.frame_count} frames from {self.video_path}")
            print(self.staleness.summary())

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

            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1

            worker.submit(frame.copy(), self.frame_count)
            result = worker.get_result()
            now = time.perf_counter()
            self.staleness.record(self.frame_count, result.frame_id, result.captured_at)

            boxes = self.extrapolator.extrapolate(result, now)
            self.renderer.draw(frame, boxes)
            cv2.imshow(self.WINDOW_NAME, frame)

            if pacer.wait(iteration_start) == self.QUIT_KEY:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Track players in a football video")
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Path to the input video (default: {DEFAULT_VIDEO})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Place a fixed-camera football video there, or pass a path as an argument.")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    PlayerTracker(video_path, model).run()


if __name__ == "__main__":
    main()
