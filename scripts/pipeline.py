import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.analytics import MatchAnalytics, PitchProjector
from scripts.ball import BallAnalytics
from scripts.calibration import (
    CalibrationWorker,
    HomographyPropagator,
    HomographyWorker,
    PitchCalibrator,
)
from scripts.display import (
    BallRenderer,
    DisplaySmoother,
    DisplayStats,
    FadeController,
    FfmpegOutput,
    FfplayViewer,
    FpsOverlay,
    FramePacer,
    MarkerRenderer,
    PitchMinimap,
    PitchOverlayRenderer,
    PlaybackDelay,
    ResultTimeline,
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
BALL_CLASS_ID = 32  # COCO class id for "sports ball"
# Ball candidates are kept down to this confidence: on match_4/match_5 the ball is
# a 10-17px blob the detector sees at a median confidence of 0.12-0.20 (docs/PLAN.md
# Plan 3.2), far below anything BoT-SORT would start a track for. Persons are
# unaffected -- BoT-SORT applies its own thresholds (botsort_custom.yaml).
DETECTION_CONF = 0.05
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
# Frames wider than this are shrunk once, right after decode, and everything --
# detection, identity, drawing, the window, the output -- runs at that size
# (see FrameScaler). 1920 leaves every current 1080p clip exactly as it was.
DISPLAY_WIDTH = 1920

# BoT-SORT's camera-motion compensation (sparse optical flow) runs on the frame
# shrunk by this factor; ultralytics hardcodes 2, and botsort.yaml cannot set it.
# On 1080p match_5 frames: 4.5ms -> 2.5ms per cycle, the estimated shift within
# 0.13px mean (0.7px max) of downscale 2 -- nothing at player-box scale.
GMC_DOWNSCALE = 4

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


class FrameScaler:
    """Shrinks each source frame once, right after decode, to the working size
    everything else runs at: detection, identity, drawing, the window, the
    output. Measured on a 4K copy of match_5: cv2.waitKey repaints a 3840px
    frame in ~30ms (1080p: ~18ms), more than the whole 20ms budget before any
    tracking runs, while the resize costs 0.4ms. Detection loses nothing --
    the model sees 1152px wide either way. Only calibration keyframes keep
    the source frame (see CalibrationWorker.submit), so their homography is
    rescaled into working pixels here. A source already at or under the
    working width passes through untouched.
    """

    def __init__(self, source_width: int, source_height: int, max_width: int):
        self.scale = min(1.0, max_width / source_width) if source_width else 1.0
        self.size = (round(source_width * self.scale), round(source_height * self.scale))

    @property
    def active(self) -> bool:
        return self.scale < 1.0

    def to_working(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        # INTER_LINEAR, not INTER_AREA: identical on the exact 2:1 of 4K ->
        # 1080p (it samples each 2x2 block's centre) at 0.4ms vs 5.3ms.
        return cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)

    def homography_to_working(self, homography: np.ndarray) -> np.ndarray:
        """Pitch -> source pixels becomes pitch -> working pixels."""
        return np.diag([self.scale, self.scale, 1.0]) @ homography


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
        classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=DETECTION_CONF,
        imgsz=INFERENCE_IMGSZ,
        verbose=False,
    )[0]


class BallCandidateCapture:
    """Grabs the raw "sports ball" detections of each forward pass.

    `model.track()` only returns *tracked* boxes, and BoT-SORT never starts a
    track for a 0.1-confidence blob, which is what the ball usually is. The
    predictor runs its `on_predict_postprocess_end` callbacks in registration
    order and the tracker's callback replaces the results in place, so a
    callback registered *before* the first `track()` call (i.e. before the
    tracker registers its own) sees every raw detection of the pass. Nothing
    extra is inferred; the ball comes out of the pass we already pay for.
    """

    def __init__(self):
        self.latest: list = []  # (x1, y1, x2, y2, conf) in pixels, this pass

    def install(self, model: YOLO) -> "BallCandidateCapture":
        model.add_callback("on_predict_postprocess_end", self)
        return self

    def __call__(self, predictor) -> None:
        result = predictor.results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.latest = []
            return
        cls = boxes.cls.cpu().numpy().astype(int)
        keep = cls == BALL_CLASS_ID
        if not keep.any():
            self.latest = []
            return
        xyxy = boxes.xyxy.cpu().numpy()[keep]
        conf = boxes.conf.cpu().numpy()[keep]
        self.latest = [
            (float(x1), float(y1), float(x2), float(y2), float(c))
            for (x1, y1, x2, y2), c in zip(xyxy, conf, strict=True)
        ]


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
    # The tracker also sees (and may track) the ball class; players only here.
    cls = boxes_obj.cls.cpu().numpy().astype(int)
    return [
        (int(x1), int(y1), int(x2), int(y2), int(tid))
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls, strict=True)
        if c == PERSON_CLASS_ID
    ]


@dataclass
class RawDetections:
    """What the inference thread publishes: this cycle's BoT-SORT boxes and the
    frame they came from (identity resolution samples jersey colors off it)."""

    boxes: list
    frame: np.ndarray | None
    frame_id: int | None
    captured_at: float | None
    ball_candidates: list = field(default_factory=list)  # (x1, y1, x2, y2, conf) px


@dataclass
class InferenceResult:
    boxes: list
    frame_id: int
    captured_at: float
    previous_boxes: list
    previous_captured_at: float | None
    coasting_progress: dict = None  # track_id -> fraction (0-1) through its grace window
    ball_candidates: list = field(default_factory=list)


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
    place player boxes on the exact frame being shown, between or ahead of the
    most recent results (see ResultTimeline).

    This thread does *only* inference and box extraction. Everything that turns
    raw BoT-SORT boxes into players (IdentityResolver) runs on the consumer's
    thread: on the 10s match_5 fixture inference alone averages 19.9ms of a 20ms
    budget, and the ~0.7ms of identity work pushed the drop rate from 4.8% to
    7.7% here, while the display thread idles ~13ms per frame.
    """

    def __init__(self, model: YOLO, ball_capture: BallCandidateCapture | None = None):
        self.model = model
        self.ball_capture = ball_capture
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

            ball_candidates = self.ball_capture.latest if self.ball_capture is not None else []
            with self.lock:
                self.latest = RawDetections(
                    boxes=boxes,
                    frame=frame,
                    frame_id=frame_id,
                    captured_at=captured_at,
                    ball_candidates=ball_candidates,
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
            ball_candidates=raw.ball_candidates,
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
        display_width: int = DISPLAY_WIDTH,
        display_delay_ms: float = 0,
        viewer: str = "opencv",
        show_markers: bool = False,
    ):
        self.video_source = video_source
        self.model = model
        # Must be installed before the first track() call (the warm-up), so it
        # runs ahead of the tracker's own callback -- see BallCandidateCapture.
        self.ball_capture = BallCandidateCapture().install(model)
        self.show_window = show_window
        self.output_path = output_path
        self.realtime = realtime
        self.analytics_dir = analytics_dir
        self.display_width = display_width
        self.display_delay_ms = display_delay_ms
        self.viewer_kind = viewer  # "opencv" (cv2.imshow) or "ffplay" (FfplayViewer)
        self.viewer = None
        self.scaler = None  # FrameScaler, created once the source size is known
        self.analytics = None  # MatchAnalytics, created once the video's fps is known
        self.ball = None  # BallAnalytics, likewise
        self.writer = None
        self.classifier = TeamClassifier()
        self.resolver = IdentityResolver(self.classifier)
        self.renderer = MarkerRenderer(self.classifier, show_markers=show_markers)
        self.pitch_overlay = PitchOverlayRenderer()
        self.minimap = PitchMinimap()
        self.ball_renderer = BallRenderer()
        self.timeline = ResultTimeline()
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
        # (frame_id, homography) as propagation produces them, so a delayed
        # frame is drawn with its own camera pose, not the newest one.
        self.homography_history = deque(maxlen=64)
        self.homography_worker = HomographyWorker(HomographyPropagator())
        self._last_keyframe_id = None

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self.scaler = FrameScaler(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.display_width,
        )
        playback = PlaybackDelay(round(self.display_delay_ms / pacer.frame_budget_ms))
        self.analytics = MatchAnalytics(fps=1000 / pacer.frame_budget_ms)
        self.ball = BallAnalytics(fps=1000 / pacer.frame_budget_ms)
        if self.output_path:
            self.writer = self._open_writer(pacer.frame_budget_ms)
        if self.show_window and self.viewer_kind == "ffplay":
            self.viewer = FfplayViewer(
                self.WINDOW_NAME, *self.scaler.size, 1000 / pacer.frame_budget_ms
            )
        self._warmup()

        worker = InferenceWorker(self.model, self.ball_capture).start()
        calibration_worker = CalibrationWorker(PitchCalibrator).start()
        self.homography_worker.start()
        self.play_start = time.perf_counter()
        try:
            self._play(cap, worker, calibration_worker, pacer, playback)
        finally:
            worker.stop()
            calibration_worker.stop()
            self.homography_worker.stop()
            cap.release()
            if self.writer is not None:
                self.writer.release()
            if self.viewer is not None:
                self.viewer.release()
            elif self.show_window:
                cv2.destroyAllWindows()
            elapsed = time.perf_counter() - self.play_start
            self.achieved_fps = self.frame_count / elapsed
            print(f"Read {self.frame_count} frames from {self.video_source}")
            print(f"Display FPS: {self.achieved_fps:.1f}")
            print(
                f"Working size: {self.scaler.size[0]}x{self.scaler.size[1]}"
                f"{' (downscaled from the source)' if self.scaler.active else ''}; "
                f"display delay: {playback.delay_frames} frames"
            )
            print()
            print(worker.stats.summary())
            print()
            print(self.resolver.stats.summary())
            print(self.resolver.splits.summary())
            print(self.resolver.identity.summary())
            print()
            print(self.staleness.summary())
            print(self.timeline.summary())
            print()
            print(self.display_stats.summary())
            print()
            self.analytics.finish()
            print(self.analytics.summary())
            print()
            print(self.ball.summary(team_name=self.classifier.team_name))
            if self.analytics_dir:
                self.analytics.write(self.analytics_dir, team_color=self._team_color)
                self.ball.write(self.analytics_dir, team_name=self.classifier.team_name)
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
            print(self.homography_worker.summary())

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
        # The tracker exists from that first call on (persist=True keeps it).
        for tracker in self.model.predictor.trackers:
            tracker.gmc.downscale = GMC_DOWNSCALE

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

        Calibration runs on the source frame (see CalibrationWorker.submit)
        while propagation runs on working-size frames, so a new keyframe's
        frame and homography are brought to working size here.
        """
        keyframe = calibration_worker.get_keyframe()
        if keyframe is not None:
            self.latest_calibration, keyframe_frame = keyframe
            is_new_keyframe = self.latest_calibration.frame_id != self._last_keyframe_id
            if is_new_keyframe and self.latest_calibration.homography is not None:
                self.homography_worker.reset(
                    self.scaler.to_working(keyframe_frame),
                    self.scaler.homography_to_working(self.latest_calibration.homography),
                    self.latest_calibration.frame_id,
                )
                self._last_keyframe_id = self.latest_calibration.frame_id

        self.homography_worker.submit(frame, self.frame_count)
        self.current_homography, homography_frame_id = self.homography_worker.get_latest()
        if homography_frame_id is not None and (
            not self.homography_history or homography_frame_id > self.homography_history[-1][0]
        ):
            self.homography_history.append((homography_frame_id, self.current_homography))

    def _homography_for(self, frame_id: int) -> np.ndarray | None:
        """The homography propagated for `frame_id` (or the nearest frame
        before it), for drawing a delayed frame with its own camera pose."""
        for history_frame_id, homography in reversed(self.homography_history):
            if history_frame_id <= frame_id:
                return homography
        return self.current_homography

    def _ball_candidates_on_pitch(self, candidates) -> list:
        """(x1, y1, x2, y2, conf) pixel candidates -> (x_m, y_m, conf) on the
        pitch, using the box bottom as the ball's contact point; off-pitch and
        uncalibrated ones are dropped."""
        projector = PitchProjector(self.current_homography)
        if not projector.available:
            return []
        on_pitch = []
        for x1, _y1, x2, y2, conf in candidates:
            position = projector.to_pitch((x1 + x2) / 2, y2)
            if position is not None:
                on_pitch.append((position[0], position[1], conf))
        return on_pitch

    def _distance_captions(self, boxes) -> dict:
        captions = {}
        for *_, track_id in boxes:
            distance = self.analytics.distance_of(track_id)
            if distance is not None:
                captions[track_id] = f"{distance:.0f}m"
        return captions

    def _team_color(self, team: int):
        return self.classifier.team_color(team) if self.classifier.centers is not None else None

    def _open_writer(self, frame_budget_ms: float) -> FfmpegOutput:
        width, height = self.scaler.size
        return FfmpegOutput(self.output_path, width, height, 1000 / frame_budget_ms)

    def _play(
        self,
        cap: cv2.VideoCapture,
        worker: InferenceWorker,
        calibration_worker: CalibrationWorker,
        pacer: FramePacer,
        playback: PlaybackDelay,
    ):
        # The display thread has ~13ms of slack per frame at 50fps (see
        # DisplayStats), so resolving identities here costs playback nothing,
        # whereas on the worker it came straight out of the inference budget.
        source_ended = False
        while True:
            iteration_start = time.perf_counter()
            self.fps_overlay.tick(iteration_start)

            t0 = time.perf_counter()
            if not source_ended:
                ret, source_frame = cap.read()
                source_ended = not ret
            t1 = time.perf_counter()

            if not source_ended:
                self.frame_count += 1
                frame = self.scaler.to_working(source_frame)
                # Read-only for both workers; `frame` itself gets drawn on.
                detection_frame = frame.copy()
                worker.submit(detection_frame, self.frame_count)
                if calibration_worker.is_keyframe(self.frame_count):
                    calibration_worker.submit(source_frame.copy(), self.frame_count)
                self._update_calibration(calibration_worker, detection_frame)
                playback.push(frame, self.frame_count)
            result = self.resolver.resolve(worker.get_result())
            self.timeline.add(result)
            # Once per new worker result (record() ignores repeats), in video
            # time, from the confirmed boxes -- not the interpolated/smoothed
            # ones drawn on screen, which are display estimates.
            self.analytics.record(
                result.frame_id,
                result.boxes,
                self.resolver.identity.occluded_player_ids,
                self.current_homography,
                self.classifier.team_for,
            )
            self.ball.record(
                result.frame_id,
                self._ball_candidates_on_pitch(result.ball_candidates),
                self.analytics.latest_positions,
                self.classifier.team_for,
            )
            t2 = time.perf_counter()

            shown = playback.pop(flush=source_ended)
            if shown is None and source_ended:
                break
            t3 = t4 = t2
            if shown is not None:
                frame, frame_id = shown
                self.staleness.record(frame_id, result.frame_id, result.captured_at)
                self._draw(frame, frame_id, pacer)
                t3 = time.perf_counter()
                if self.viewer is not None:
                    self.viewer.write(frame)
                elif self.show_window:
                    cv2.imshow(self.WINDOW_NAME, frame)
                if self.writer is not None:
                    self.writer.write(frame)
                t4 = time.perf_counter()

            if self.realtime:
                key = pacer.wait(iteration_start, use_gui=self.show_window and self.viewer is None)
            else:
                key = -1
            t5 = time.perf_counter()

            if shown is not None:
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
            if self.viewer is not None and self.viewer.closed:
                break  # its window was closed

    def _draw(self, frame: np.ndarray, frame_id: int, pacer: FramePacer) -> None:
        boxes, coasting_progress, synced = self.timeline.boxes_at(frame_id)
        boxes = self.smoother.smooth(boxes, ease=not synced)
        alphas = self.fader.update(boxes, coasting_progress)
        homography = self._homography_for(frame_id)
        self.pitch_overlay.draw(frame, homography)
        self.renderer.draw(frame, boxes, alphas, captions=self._distance_captions(boxes))
        self.minimap.draw(
            frame,
            self.analytics.latest_positions,
            self.classifier.team_for,
            self._team_color,
            ball=self.ball.ball,
            holder=self.ball.holder,
        )
        self.ball_renderer.draw(
            frame,
            self.ball,
            homography,
            self.analytics.latest_positions,
            team_name=self.classifier.team_name,
        )
        if self.show_window:
            self.fps_overlay.draw(frame, native_fps=1000 / pacer.frame_budget_ms)
