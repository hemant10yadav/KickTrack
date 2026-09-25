import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.analytics import MatchAnalytics
from scripts.calibration import (
    CalibrationWorker,
    HomographyPropagator,
    HomographyWorker,
    PitchCalibrator,
)
from scripts.display import (
    DisplaySmoother,
    DisplayStats,
    FadeController,
    FpsOverlay,
    FramePacer,
    MarkerRenderer,
    MotionExtrapolator,
    PitchMinimap,
    PitchOverlayRenderer,
    StalenessTracker,
)
from scripts.player import (
    JerseySampler,
    PlayerIdentityManager,
    SplitDetectionSuppressor,
    StateManager,
    TeamClassifier,
    boxes_overlap,
    extract_jersey_color,
)

STREAM_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")  # e.g. rtmp://, rtsp://, http(s)://

PERSON_CLASS_ID = 0  # COCO class id for "person"
TRACKER_CONFIG = str(Path(__file__).parent / "botsort_custom.yaml")
MODEL_NAME = "yolo26s.mlpackage"
# Switched from yolov8s (docs/PLAN.md Plan 2.9): the swaps of Plan 2.8 all began
# with the detector returning one box over two overlapping bodies, and for two of
# the three match_5 episodes v8s *did* also return the second body -- at a
# confidence BoT-SORT's new_track_thresh (0.5) never starts a track for.
# Lowering that threshold trades merges for churn (raw ids 63 -> 144 on match_5),
# and yolov8m separates no better. YOLO26s (NMS-free head) keeps two *tracked*
# boxes on 19/19 keeper+defender frames (v8s: 3/19), 34/34 blue+white frames
# (29/34) and 24/73 of the hardest pair (1/73) with the tracker config unchanged,
# mints fewer ids (46-48 vs 50-53) and passes every hand-checked identity probe.
# Visual recall on spot frames of match_4/match_5 is equal (every on-pitch player
# v8s boxes, 26s boxes; v8s's extra boxes were low-confidence duplicates).
# Cost: ~+0.4ms/cycle (predict-only 12.4 vs 11.9ms; realtime worker 20.2 vs
# 19.8ms on a loaded machine, 26 vs 23 frames dropped) -- inside run-to-run noise.
# Earlier history, still true of the size choice: v8s over v8m at this imgsz cut
# inference ~24 -> ~18ms and cleared match_5's 20ms budget; every on-pitch
# player caught by v8m was also caught by v8s (verified visually).
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
class RawDetections:
    """What the inference thread publishes: this cycle's BoT-SORT boxes and the
    frame they came from (identity resolution samples jersey colors off it)."""

    boxes: list
    frame: np.ndarray | None
    frame_id: int | None
    captured_at: float | None


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
    box extraction vs. whatever is left, plus how many submitted frames the
    latest-frame buffer ended up dropping before the worker could get to them.
    """

    def __init__(self):
        self.frames_submitted = 0
        self.frames_skipped = 0
        self.inference_ms = []
        self.extract_ms = []
        self.residual_ms = []
        self.total_ms = []

    def record_submit(self, was_pending_overwritten: bool):
        self.frames_submitted += 1
        if was_pending_overwritten:
            self.frames_skipped += 1

    def record_processed(self, inference_ms: float, extract_ms: float, total_ms: float):
        self.inference_ms.append(inference_ms)
        self.extract_ms.append(extract_ms)
        self.residual_ms.append(total_ms - inference_ms - extract_ms)
        self.total_ms.append(total_ms)

    def summary(self) -> str:
        frames_processed = len(self.total_ms)
        lines = [
            f"Frames submitted: {self.frames_submitted}",
            f"Frames skipped (overwritten before processing): {self.frames_skipped}",
            f"Frames processed by worker: {frames_processed}",
        ]
        if frames_processed:
            inf = np.array(self.inference_ms)
            lines.append(
                f"YOLO+tracker inference latency (ms): avg={inf.mean():.1f} "
                f"min={inf.min():.1f} max={inf.max():.1f}"
            )
            lines.append(f"Effective inference FPS: {1000 / inf.mean():.1f}")
            lines.append(_stat("extract_boxes (ms)", self.extract_ms))
            lines.append(_stat("residual/unaccounted (ms)", self.residual_ms))
            lines.append(_stat("Worker total latency (ms)", self.total_ms))
        return "\n".join(lines)


def _stat(name, values):
    arr = np.array(values)
    return f"{name}: avg={arr.mean():.1f} min={arr.min():.1f} max={arr.max():.1f}"


class InferenceWorker:
    """Runs detection + tracking continuously in the background, always working on
    the most recently submitted frame. Frames arriving faster than inference finishes
    are dropped (never queued), so the worker never falls further and further behind.

    Each submitted frame carries a frame_id and capture timestamp, which is echoed
    back with the result — this lets the caller measure exactly how many frames (and
    how many milliseconds) old the boxes it's currently displaying are, and also
    extrapolate player motion forward using the two most recent results (see
    MotionExtrapolator).

    This thread does *only* inference and box extraction. Everything that turns
    raw BoT-SORT boxes into players (IdentityResolver) runs on the consumer's
    thread: on the 10s match_5 fixture inference alone averages 19.9ms of a 20ms
    budget, and the ~0.7ms of identity work pushed the drop rate from 4.8% to
    7.7% here, while the display thread idles ~13ms per frame.
    """

    def __init__(self, model: YOLO):
        self.model = model
        self.lock = threading.Lock()
        self.pending = None  # (frame, frame_id, captured_at)
        self.latest = RawDetections(boxes=[], frame=None, frame_id=None, captured_at=None)
        self.running = True
        self.stats = WorkerStats()
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

    def get_result(self) -> RawDetections:
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

            total_ms = (time.perf_counter() - total_start) * 1000
            self.stats.record_processed(inference_ms, extract_ms, total_ms)

            with self.lock:
                self.latest = RawDetections(
                    boxes=boxes, frame=frame, frame_id=frame_id, captured_at=captured_at
                )

    def _take_pending(self):
        with self.lock:
            pending, self.pending = self.pending, None
            return pending


class ResolverStats:
    """Where IdentityResolver's time goes per worker cycle: split suppression +
    identity reconciliation vs. the jersey-color sampling loop."""

    def __init__(self):
        self.identity_ms = []
        self.jersey_ms = []
        self.boxes_seen = 0
        self.jersey_extractions = 0

    def record(self, identity_ms: float, jersey_ms: float):
        self.identity_ms.append(identity_ms)
        self.jersey_ms.append(jersey_ms)

    def summary(self) -> str:
        if not self.identity_ms:
            return "Identity resolution: no worker result was ever resolved"
        return "\n".join(
            [
                _stat("split suppression + identity (ms)", self.identity_ms),
                _stat("jersey extraction loop (ms)", self.jersey_ms),
                f"Jersey extractions: {self.jersey_extractions}/{self.boxes_seen} boxes seen "
                f"({100 * self.jersey_extractions / max(1, self.boxes_seen):.1f}%)",
            ]
        )


class IdentityResolver:
    """Turns one cycle of raw BoT-SORT boxes into confirmed, persistently
    identified players: split-detection suppression, track_id -> player_id
    reconciliation (PlayerIdentityManager), jersey sampling for team
    classification, and confirm/grace visibility state. Runs once per *new*
    worker result, on whichever thread consumes results -- the display loop --
    so none of it competes with inference for the worker's frame budget.
    """

    def __init__(self, classifier: TeamClassifier):
        self.classifier = classifier
        self.state = StateManager()
        self.identity = PlayerIdentityManager()
        self.splits = SplitDetectionSuppressor()
        self.stats = ResolverStats()
        self.cycle_count = 0
        self.jersey_last_sampled = {}
        self.latest = InferenceResult(
            boxes=[],
            frame_id=None,
            captured_at=None,
            previous_boxes=[],
            previous_captured_at=None,
            coasting_progress={},
        )

    def resolve(self, raw: RawDetections) -> InferenceResult:
        """Returns the resolved result for `raw`; a result already resolved (the
        worker has not produced a new one since) is returned as is."""
        if raw.frame_id is None or raw.frame_id == self.latest.frame_id:
            return self.latest
        frame = raw.frame
        self.cycle_count += 1

        identity_start = time.perf_counter()
        # Collapse one-body-two-boxes detections before anything keys an
        # identity off them, or that player is tracked and counted twice.
        boxes = self.splits.update(raw.boxes, self.cycle_count)
        # Reconcile BoT-SORT's own transient track_id into a persistent player_id
        # (see PlayerIdentityManager) before anything downstream (team
        # classification, confirm/grace display state) keys off it. The jersey
        # sampler lets it anchor each identity to the color it was seen in.
        boxes = self.identity.update(boxes, self.cycle_count, JerseySampler(frame))
        identity_ms = (time.perf_counter() - identity_start) * 1000

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
        self.stats.record(identity_ms, jersey_ms)

        confirmed_boxes = self.state.update(boxes)
        coasting_progress = {
            track_id: min(1.0, miss_count / self.state.GRACE_CYCLES)
            for track_id, miss_count in self.state.miss_counts.items()
            if miss_count > 0
        }
        self.latest = InferenceResult(
            boxes=confirmed_boxes,
            frame_id=raw.frame_id,
            captured_at=raw.captured_at,
            previous_boxes=self.latest.boxes,
            previous_captured_at=self.latest.captured_at,
            coasting_progress=coasting_progress,
        )
        return self.latest

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
    long a single inference call takes. Each new worker result is turned into
    identified players here, on the display thread (see IdentityResolver).
    """

    WINDOW_NAME = "Football Tracker"
    QUIT_KEY = ord("q")

    def __init__(
        self,
        video_source: str | int,
        model: YOLO,
        show_window: bool = True,
        output_path: str | None = None,
        realtime: bool = True,
        analytics_dir: str | None = None,
    ):
        self.video_source = video_source
        self.model = model
        self.show_window = show_window
        self.output_path = output_path
        self.realtime = realtime
        self.analytics_dir = analytics_dir
        self.analytics = None  # MatchAnalytics, created once the video's fps is known
        self.writer = None
        self.classifier = TeamClassifier()
        self.resolver = IdentityResolver(self.classifier)
        self.renderer = MarkerRenderer(self.classifier)
        self.pitch_overlay = PitchOverlayRenderer()
        self.minimap = PitchMinimap()
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
        self.current_homography = None
        self.homography_worker = HomographyWorker(HomographyPropagator())
        self._last_keyframe_id = None

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self.analytics = MatchAnalytics(fps=1000 / pacer.frame_budget_ms)
        if self.output_path:
            self.writer = self._open_writer(cap, pacer.frame_budget_ms)
        self._warmup()

        worker = InferenceWorker(self.model).start()
        calibration_worker = CalibrationWorker(PitchCalibrator).start()
        self.homography_worker.start()
        self.play_start = time.perf_counter()
        try:
            self._play(cap, worker, calibration_worker, pacer)
        finally:
            worker.stop()
            calibration_worker.stop()
            self.homography_worker.stop()
            cap.release()
            if self.writer is not None:
                self.writer.release()
            if self.show_window:
                cv2.destroyAllWindows()
            elapsed = time.perf_counter() - self.play_start
            self.achieved_fps = self.frame_count / elapsed
            print(f"Read {self.frame_count} frames from {self.video_source}")
            print(f"Display FPS: {self.achieved_fps:.1f}")
            print()
            print(worker.stats.summary())
            print()
            print(self.resolver.stats.summary())
            print(self.resolver.splits.summary())
            print(self.resolver.identity.summary())
            print()
            print(self.staleness.summary())
            print()
            print(self.display_stats.summary())
            print()
            self.analytics.finish()
            print(self.analytics.summary())
            if self.analytics_dir:
                self.analytics.write(self.analytics_dir, team_color=self._team_color)
                print(f"Analytics written to {self.analytics_dir}/")
            print()
            if self.latest_calibration is not None:
                print(
                    f"Calibration: keyframe_homography_found="
                    f"{self.latest_calibration.homography is not None} "
                    f"keypoints={self.latest_calibration.num_keypoints} "
                    f"lines={self.latest_calibration.num_lines} "
                    f"propagation_active={self.current_homography is not None}"
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

    def _update_calibration(self, calibration_worker: CalibrationWorker, frame: np.ndarray):
        """Forwards each new full recalibration from CalibrationWorker
        (keyframe rate) to HomographyWorker's reset(), and submits every
        displayed frame for propagation -- both calls return immediately;
        HomographyWorker does the actual optical-flow tracking on its own
        background thread. See docs/PITCH_CALIBRATION_SPEC.md Phase 2: this
        used to call HomographyPropagator directly here, synchronously,
        which pushed ~9-10% of frames over budget on match_5's tight 20ms
        window even though the work itself only cost ~1-3ms -- moving it
        off-thread removes it from the budget entirely.
        """
        keyframe = calibration_worker.get_keyframe()
        if keyframe is not None:
            self.latest_calibration, keyframe_frame = keyframe
            is_new_keyframe = self.latest_calibration.frame_id != self._last_keyframe_id
            if is_new_keyframe and self.latest_calibration.homography is not None:
                self.homography_worker.reset(
                    keyframe_frame,
                    self.latest_calibration.homography,
                    self.latest_calibration.frame_id,
                )
                self._last_keyframe_id = self.latest_calibration.frame_id

        self.homography_worker.submit(frame, self.frame_count)
        self.current_homography, _ = self.homography_worker.get_latest()

    def _distance_captions(self, boxes) -> dict:
        captions = {}
        for *_, track_id in boxes:
            distance = self.analytics.distance_of(track_id)
            if distance is not None:
                captions[track_id] = f"{distance:.0f}m"
        return captions

    def _team_color(self, team: int):
        return self.classifier.team_color(team) if self.classifier.centers is not None else None

    def _open_writer(self, cap: cv2.VideoCapture, frame_budget_ms: float) -> cv2.VideoWriter:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(self.output_path, fourcc, 1000 / frame_budget_ms, (width, height))

    def _play(
        self,
        cap: cv2.VideoCapture,
        worker: InferenceWorker,
        calibration_worker: CalibrationWorker,
        pacer: FramePacer,
    ):
        # The display thread has ~13ms of slack per frame at 50fps (see
        # DisplayStats), so resolving identities here costs playback nothing,
        # whereas on the worker it came straight out of the inference budget.
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
            if calibration_worker.is_keyframe(self.frame_count):
                calibration_worker.submit(frame.copy(), self.frame_count)
            result = self.resolver.resolve(worker.get_result())
            self._update_calibration(calibration_worker, frame)
            # Once per new worker result (record() ignores repeats), in video
            # time, from the confirmed boxes -- not the extrapolated/smoothed
            # ones drawn on screen, which are display estimates.
            self.analytics.record(
                result.frame_id,
                result.boxes,
                self.resolver.identity.occluded_player_ids,
                self.current_homography,
                self.classifier.team_for,
            )
            t2 = time.perf_counter()
            now = time.perf_counter()
            self.staleness.record(self.frame_count, result.frame_id, result.captured_at)

            boxes = self.extrapolator.extrapolate(result, now)
            boxes = self.smoother.smooth(boxes)
            alphas = self.fader.update(boxes, result.coasting_progress)
            self.pitch_overlay.draw(frame, self.current_homography)
            self.renderer.draw(frame, boxes, alphas, captions=self._distance_captions(boxes))
            self.minimap.draw(
                frame, self.analytics.latest_positions, self.classifier.team_for, self._team_color
            )
            t3 = time.perf_counter()
            if self.show_window:
                self.fps_overlay.draw(frame, native_fps=1000 / pacer.frame_budget_ms)
                cv2.imshow(self.WINDOW_NAME, frame)
            if self.writer is not None:
                self.writer.write(frame)
            t4 = time.perf_counter()

            if self.realtime:
                key = pacer.wait(iteration_start, use_gui=self.show_window)
            else:
                key = -1
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
