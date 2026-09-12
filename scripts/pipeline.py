import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.calibration import CalibrationWorker, PitchCalibrator
from scripts.display import (
    DisplaySmoother,
    DisplayStats,
    FadeController,
    FpsOverlay,
    FramePacer,
    MarkerRenderer,
    MotionExtrapolator,
    StalenessTracker,
)
from scripts.player import (
    PlayerIdentityManager,
    StateManager,
    TeamClassifier,
    boxes_overlap,
    extract_jersey_color,
)

STREAM_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")  # e.g. rtmp://, rtsp://, http(s)://

PERSON_CLASS_ID = 0  # COCO class id for "person"
TRACKER_CONFIG = str(Path(__file__).parent / "botsort_custom.yaml")
MODEL_NAME = "yolov8s.mlpackage"
# Switched from yolov8m: at the same 640x1152 rect imgsz, v8s cuts inference from
# ~24ms to ~18ms avg on match_5.mp4 (50fps, 20ms budget) -- enough to clear the
# budget outright instead of just narrowing the shortfall, eliminating match_5's
# frame drops entirely (was 33% before the device="mps" fix, ~19-27% after it,
# 0% with v8s). Verified visually (not just box-count) across match_4/match_5 spot
# frames: every on-pitch player caught by v8m was also caught by v8s; box-count
# differences were sideline/crowd false positives, same pattern already seen with
# the rect-imgsz change above -- never a missed player.
# (height, width): matches the 16:9 aspect ratio of the actual footage instead of
# padding to a square, and is smaller than the original 1280 square export.
# Verified visually (not just by box-count) on both a 1080p 50fps clip
# (match_5.mp4) and a 4K 30fps clip (match_4.mp4): every player detected at
# 736x1280 is still detected here at equal-or-better confidence -- box count on
# spot-check frames was 33-34 either way, only the padding pixels were cut, not
# player detail. Measured latency: ~35ms -> ~18ms/frame (about 2x), which gets a
# 50fps source under its 20ms native frame budget for the first time (previously
# inference was always slower than 50fps playback, so the async worker was
# structurally guaranteed to drop frames no matter how it was tuned).
# The exported .mlpackage has this shape baked into its input tensor, so this must
# stay in sync with the `imgsz` used at export time (see CLAUDE.md export command).
INFERENCE_IMGSZ = (640, 1152)

JERSEY_RESAMPLE_INTERVAL = 15  # worker cycles between re-observations of a settled track


def resolve_video_source(raw: str) -> str | int:
    """Turn a CLI argument into whatever cv2.VideoCapture should receive.

    A live source (stream URL or webcam device index) must bypass Path entirely:
    Path() silently collapses a URL's "://" down to ":/" (confirmed: Path("rtmp://
    host/live") -> "rtmp:/host/live"), which cv2.VideoCapture then fails to open.
    A plain local file still goes through Path so relative paths resolve the same
    way they always have.
    """
    if raw.isdigit():
        return int(raw)  # webcam device index, e.g. "0"
    if STREAM_SCHEME_RE.match(raw):
        return raw  # stream URL, passed through untouched
    return str(Path(raw))


def is_stream_source(source: str | int) -> bool:
    return isinstance(source, int) or bool(STREAM_SCHEME_RE.match(source))


def track(model: YOLO, frame):
    """Run detection + tracking on a single frame and return raw Ultralytics results.

    No `device` kwarg: the CoreML backend always runs on the Neural Engine
    (ComputeUnit.CPU_AND_NE, set in its own load_model) regardless of what's passed
    here -- a leftover `device="mps"` from before the CoreML migration was still
    read by the predictor, which moved the input tensor onto the MPS GPU and back
    to CPU every frame before the real CoreML call, for no benefit. Measured cost of
    that round trip: ~4.6ms/frame on match_5.mp4 (29.2ms -> 24.6ms avg inference),
    dropping its frame-drop rate from 33% to 19.5%.
    """
    return model.track(
        frame,
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker=TRACKER_CONFIG,
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
        self.identity = PlayerIdentityManager()
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
            # Reconcile BoT-SORT's own transient track_id into a persistent player_id
            # (see PlayerIdentityManager) before anything downstream (team
            # classification, confirm/grace display state) keys off it.
            boxes = self.identity.update(boxes, self.cycle_count)

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


class PlayerTracker:
    """Plays a video live, overlaying player tracking markers.

    Detection/tracking runs on a background thread (see InferenceWorker) so
    display always paces at the video's real frame rate, independent of how
    long a single inference call takes.
    """

    WINDOW_NAME = "Football Tracker"
    QUIT_KEY = ord("q")

    def __init__(self, video_source: str | int, model: YOLO, show_window: bool = True):
        self.video_source = video_source
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
        self.latest_calibration = None

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self._warmup()

        worker = InferenceWorker(self.model, self.classifier).start()
        calibration_worker = CalibrationWorker(PitchCalibrator()).start()
        self.play_start = time.perf_counter()
        try:
            self._play(cap, worker, calibration_worker, pacer)
        finally:
            worker.stop()
            calibration_worker.stop()
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            elapsed = time.perf_counter() - self.play_start
            self.achieved_fps = self.frame_count / elapsed
            print(f"Read {self.frame_count} frames from {self.video_source}")
            print(f"Display FPS: {self.achieved_fps:.1f}")
            print()
            print(worker.stats.summary())
            print()
            print(worker.identity.summary())
            print()
            print(self.staleness.summary())
            print()
            print(self.display_stats.summary())
            print()
            if self.latest_calibration is not None:
                homography_found = self.latest_calibration.homography is not None
                print(
                    f"Calibration: homography_found={homography_found} "
                    f"keypoints={self.latest_calibration.num_keypoints} "
                    f"lines={self.latest_calibration.num_lines}"
                )
            else:
                print("Calibration: no result yet")

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Could not open video: {self.video_source}")
            sys.exit(1)
        return cap

    def _warmup(self):
        """Pay the one-time GPU kernel compilation cost (MPS) before playback starts."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track(self.model, blank_frame)

    def _play(
        self,
        cap: cv2.VideoCapture,
        worker: InferenceWorker,
        calibration_worker: CalibrationWorker,
        pacer: FramePacer,
    ):
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
            calibration_worker.submit(frame.copy(), self.frame_count)
            result = worker.get_result()
            self.latest_calibration = calibration_worker.get_result()
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
