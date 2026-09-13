import math
import time

import cv2
import numpy as np

from scripts.calibration import PITCH_LINES
from scripts.player import TeamClassifier

UNCLASSIFIED_COLOR = (180, 180, 180)  # gray, shown before a track has enough samples


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


class PitchOverlayRenderer:
    """Draws the pitch's standard line markings (PITCH_LINES, in
    scripts/calibration.py) reprojected through the current pitch
    homography, so calibration is visible on the live video instead of
    only reported as end-of-run stats (see docs/PITCH_CALIBRATION_SPEC.md).

    Coordinates far outside the frame (a homography extrapolated well
    beyond where it was actually fit -- see the Phase 2 drift-measurement
    write-up on why that's numerically unstable) are clamped before
    drawing so a bad calibration can't overflow OpenCV's int line drawing.
    """

    LINE_COLOR = (0, 0, 255)  # red, distinct from team marker colors
    LINE_THICKNESS = 2
    CLAMP_MARGIN = 10_000  # px beyond the frame edge; well past anything meaningful

    def draw(self, frame, homography):
        if homography is None:
            return
        h, w = frame.shape[:2]
        for (x1, y1), (x2, y2) in PITCH_LINES:
            p1 = self._project(homography, x1, y1, w, h)
            p2 = self._project(homography, x2, y2, w, h)
            if p1 is None or p2 is None:
                continue
            cv2.line(frame, p1, p2, self.LINE_COLOR, self.LINE_THICKNESS)

    def _project(self, homography, x, y, w, h):
        point = homography @ np.array([x, y, 1.0])
        if abs(point[2]) < 1e-6:
            return None
        point /= point[2]
        px = int(np.clip(point[0], -self.CLAMP_MARGIN, w + self.CLAMP_MARGIN))
        py = int(np.clip(point[1], -self.CLAMP_MARGIN, h + self.CLAMP_MARGIN))
        return px, py


class MotionExtrapolator:
    """Shifts each player's last known box forward in time using the velocity
    estimated between the two most recent AI results, so the displayed marker keeps
    moving smoothly between inference updates instead of freezing at a stale position.

    A max shift cap guards against runaway extrapolation from a noisy velocity
    estimate (e.g. an ID that just switched, or a very short dt between results).
    """

    def __init__(self, max_shift_px=120):
        self.max_shift_px = max_shift_px

    def extrapolate(self, result: "InferenceResult", now: float):  # noqa: F821 -- pipeline.InferenceResult; string to avoid a circular import (pipeline.py imports this module)
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
