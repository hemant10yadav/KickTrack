import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from scripts.analytics import PITCH_LENGTH_M, PITCH_WIDTH_M, PitchProjector, render_pitch
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

    def __init__(self, classifier: TeamClassifier, show_markers: bool = False):
        self.classifier = classifier
        self.show_markers = show_markers  # pin + ID label; off leaves only the caption
        self.anchor_positions = {}  # track_id -> (tip_x, tip_y) floats

    def draw(self, frame, boxes, alphas=None, captions=None):
        """captions: optional track_id -> short text drawn under the ID label
        (the running distance covered, in the live view)."""
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
            caption = None if captions is None else captions.get(track_id)
            if self.show_markers:
                self._draw_marker(frame, tip, bbox_height, color, alpha)
                self._draw_label(frame, x1, tip, bbox_height, track_id, color, caption)
            elif caption:
                self._draw_caption(frame, x1, tip[1], caption)
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

    def _draw_label(self, frame, x1, tip, bbox_height, track_id, color, caption=None):
        half_width = int(
            np.clip(
                bbox_height * self.HALF_WIDTH_FROM_HEIGHT, self.MIN_HALF_WIDTH, self.MAX_HALF_WIDTH
            )
        )
        pin_height = int(half_width * self.PIN_HEIGHT_RATIO)
        baseline = max(0, tip[1] - pin_height - 6)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        if caption:
            # Above the ID so the pin keeps its clear gap to the player's head.
            self._draw_caption(frame, x1, baseline - 16, caption)

    def _draw_caption(self, frame, x, baseline, caption):
        # White so it reads on every kit colour.
        cv2.putText(
            frame,
            caption,
            (x, max(0, baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


class PitchMinimap:
    """A small top-down pitch in a corner of the live frame with a dot per
    player at their calibrated pitch position (from MatchAnalytics'
    latest_positions -- the same gated samples the heatmaps are built from),
    coloured by team and labelled with the player's id so a dot can be matched
    to the marker on the video. The pitch itself is rendered once and blended
    in per frame; only the dots are drawn per frame.
    """

    SCALE = 3  # px per metre: 315x204 for a 105x68 pitch
    MARGIN = 12
    ALPHA = 0.85
    DOT_RADIUS = 4

    def __init__(self):
        self.pitch = render_pitch(self.SCALE)

    BALL_COLOR = (0, 255, 255)

    def draw(self, frame, positions: dict, team_of, team_color, ball=None, holder=None) -> None:
        """positions: player_id -> (x_m, y_m); team_of(id) -> team or None;
        team_color(team) -> BGR or None; ball: BallState or None; holder: the
        player_id in possession, ringed."""
        h, w = frame.shape[:2]
        ph, pw = self.pitch.shape[:2]
        if ph + 2 * self.MARGIN > h or pw + 2 * self.MARGIN > w:
            return
        panel = self.pitch.copy()
        for player_id, (x, y) in positions.items():
            px = int((x + PITCH_LENGTH_M / 2) * self.SCALE)
            py = int((y + PITCH_WIDTH_M / 2) * self.SCALE)
            if not (0 <= px < pw and 0 <= py < ph):
                continue
            team = team_of(player_id)
            color = team_color(team) if team is not None else None
            color = UNCLASSIFIED_COLOR if color is None else color
            if player_id == holder:
                cv2.circle(
                    panel, (px, py), self.DOT_RADIUS + 4, self.BALL_COLOR, 2, lineType=cv2.LINE_AA
                )
            cv2.circle(panel, (px, py), self.DOT_RADIUS + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, (px, py), self.DOT_RADIUS, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                panel,
                str(player_id),
                (px + self.DOT_RADIUS + 1, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        if ball is not None:
            bx = int((ball.x + PITCH_LENGTH_M / 2) * self.SCALE)
            by = int((ball.y + PITCH_WIDTH_M / 2) * self.SCALE)
            if 0 <= bx < pw and 0 <= by < ph:
                cv2.circle(panel, (bx, by), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(panel, (bx, by), 2, self.BALL_COLOR, -1, lineType=cv2.LINE_AA)
        if not positions:
            cv2.putText(
                panel,
                "no calibrated positions",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        y0, x0 = self.top(frame), w - self.MARGIN - pw
        region = frame[y0 : y0 + ph, x0 : x0 + pw]
        cv2.addWeighted(panel, self.ALPHA, region, 1 - self.ALPHA, 0, dst=region)
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (255, 255, 255), 1)

    def top(self, frame) -> int:
        """The minimap's top edge in `frame`, so a panel can sit on it."""
        return frame.shape[0] - self.MARGIN - self.pitch.shape[0]


class TeamShapeOverlay:
    """Each team's outline and defensive line drawn on the pitch in the video.

    The outline goes through the feet of the shape's players in the boxes
    shown on this very frame (ResultTimeline's), not their positions at the
    last shape sample, so it sits on them like the markers do; a projection
    keeps a convex outline convex, so the outline of the feet on screen is
    the pitch outline seen by the camera. The defensive line runs
    touchline to touchline on the grass at the team's last outfield man,
    found on this frame too -- the same feet taken to the pitch through the
    frame's own homography -- so it moves with him; drawn from the 0.1 s
    shape samples, smoothed for the panel, it trailed a stepping-up defence.
    It is drawn while the line is measurable (the team's own-goal end in
    view, see TeamShapeAnalytics), since otherwise the deepest player seen
    need not be the last man.
    Thin lines only, so the players stay the thing to look at (a faint tint
    inside was tried: blending its bounding box of a 1920-wide frame cost
    0.65 ms per team, for a tint hardly seen); the second team's lines are
    dashed, because two pale kits (sky blue and white on match_5) otherwise
    read as the same outline."""

    OUTLINE_PX = 1
    LINE_PX = 2
    DASHED_TEAM = 1
    DASH_PX, GAP_PX = 8, 6

    def __init__(self):
        self._projector = PitchOverlayRenderer()

    def draw(self, frame, boxes, shape_analytics, homography, team_color) -> None:
        feet = {track_id: ((x1 + x2) / 2, y2) for x1, _y1, x2, y2, track_id in boxes}
        h, w = frame.shape[:2]
        projector = PitchProjector(homography)
        for team, shape in shape_analytics.latest.items():
            color = team_color(team)
            if color is None:
                continue
            points = np.array([feet[pid] for pid in shape.player_ids if pid in feet], np.float32)
            dashed = team == self.DASHED_TEAM
            if len(points) >= 3:
                self._draw_outline(frame, cv2.convexHull(points).astype(np.int32), color, dashed)
            x = self._last_man_x(team, shape, shape_analytics, feet, projector)
            if x is not None:
                a = self._projector._project(homography, x, -PITCH_WIDTH_M / 2, w, h)
                b = self._projector._project(homography, x, PITCH_WIDTH_M / 2, w, h)
                if a is not None and b is not None:
                    self._segment(frame, a, b, color, self.LINE_PX, dashed)

    @staticmethod
    def _last_man_x(team, shape, shape_analytics, feet, projector) -> float | None:
        """Pitch x of the team's player nearest its own goal on this frame."""
        own_goal = shape_analytics.own_goal
        if own_goal is None or not projector.available:
            return None
        if shape_analytics.display_value(team, "line_m") is None:
            return None
        xs = [
            xy[0]
            for xy in (projector.to_pitch(*feet[pid]) for pid in shape.player_ids if pid in feet)
            if xy is not None
        ]
        return max(xs, key=lambda x: own_goal[team] * x) if xs else None

    def _draw_outline(self, frame, hull, color, dashed: bool) -> None:
        corners = hull.reshape(-1, 2)
        self._polyline(frame, np.vstack([corners, corners[:1]]), color, self.OUTLINE_PX, dashed)

    def _segment(self, frame, a, b, color, thickness: int, dashed: bool) -> None:
        self._polyline(frame, np.array([a, b]), color, thickness, dashed)

    def _polyline(self, frame, points: np.ndarray, color, thickness: int, dashed: bool) -> None:
        """Solid, or dashed: every dash of every edge is computed at once and
        drawn in one cv2.polylines call -- a cv2.line per dash from Python
        (~150 per outline) took the draw from 2.5 to 6 ms a frame on match_5
        and the worker's skipped frames from 34 to 481."""
        points = points.astype(np.float64)
        if not dashed:
            cv2.polylines(frame, [points.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
            return
        a, b = points[:-1], points[1:]
        lengths = np.hypot(*(b - a).T)
        period = self.DASH_PX + self.GAP_PX
        dashes = []
        for start, end, length in zip(a, b, lengths, strict=True):
            if length < 1:
                continue
            step = (end - start) / length
            d = np.arange(0.0, length, period)[:, None]
            dashes.append(
                np.stack(
                    [start + step * d, start + step * np.minimum(d + self.DASH_PX, length)], axis=1
                )
            )
        if dashes:
            segments = np.concatenate(dashes)
            h, w = frame.shape[:2]
            inside = (np.abs(segments[:, :, 0] - w / 2) < w) & (
                np.abs(segments[:, :, 1] - h / 2) < h
            )
            segments = segments[inside.any(axis=1)]
            cv2.polylines(
                frame, list(segments.astype(np.int32)), False, color, thickness, cv2.LINE_AA
            )


class TeamShapePanel:
    """Each team's width, depth and defensive line (distance from its own
    goal), in metres, in a panel sitting on top of the minimap. A value the
    camera cannot measure right now (TeamShapeAnalytics) reads "-" rather
    than a guess."""

    WIDTH = PitchMinimap.SCALE * int(PITCH_LENGTH_M)
    LINE_PX = 24
    GAP = 6
    COLUMNS = (("width", "width_m"), ("depth", "depth_m"), ("line", "line_m"))
    COLUMN_PX = 62

    def draw(self, frame, shape_analytics, bottom: int, team_name=str, team_color=None) -> None:
        """bottom: the y the panel's lower edge sits at (PitchMinimap.top)."""
        team_color = team_color or (lambda _team: None)
        height = self.LINE_PX * 3 + 8
        w = frame.shape[1]
        x0 = w - PitchMinimap.MARGIN - self.WIDTH
        y0 = bottom - self.GAP - height
        if y0 < 0 or x0 < 0:
            return
        overlay = frame[y0 : y0 + height, x0 : x0 + self.WIDTH]
        cv2.addWeighted(np.zeros_like(overlay), 0.55, overlay, 0.45, 0, dst=overlay)
        baseline = y0 + self.LINE_PX
        self._text(frame, "Team shape", (x0 + 8, baseline), 0.5)
        for i, (label, _metric) in enumerate(self.COLUMNS):
            self._text_right(frame, label, self._column_right(x0, i), baseline, 0.45)
        for row, team in enumerate((0, 1)):
            baseline = y0 + self.LINE_PX * (row + 2)
            color = team_color(team)
            if color is not None:
                cv2.rectangle(frame, (x0 + 8, baseline - 12), (x0 + 20, baseline), color, -1)
            name = team_name(team) if color is not None else f"team {team + 1}"
            self._text(frame, name, (x0 + 26, baseline), 0.5)
            for i, (_label, metric) in enumerate(self.COLUMNS):
                value = shape_analytics.display_value(team, metric)
                text = "-" if value is None else f"{value:.0f}m"
                self._text_right(frame, text, self._column_right(x0, i), baseline, 0.5)

    def _column_right(self, x0: int, i: int) -> int:
        return x0 + self.WIDTH - 8 - (len(self.COLUMNS) - 1 - i) * self.COLUMN_PX

    @staticmethod
    def _text(frame, text, origin, scale) -> None:
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    @classmethod
    def _text_right(cls, frame, text, right, baseline, scale) -> None:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cls._text(frame, text, (right - tw, baseline), scale)


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


class BallRenderer:
    """The tracked ball on the video (a ring at its pitch position, hollow and
    dimmer while coasting through a gap), a ring around the player in
    possession, and the team pass counts in the top-right corner. The two
    rings are drawn only with show_markers, the panel unless show_passes is off."""

    BALL_COLOR = (0, 255, 255)
    PANEL_MARGIN = 12

    def __init__(self, show_markers: bool = False, show_passes: bool = True):
        self.show_markers = show_markers
        self.show_passes = show_passes

    def draw(
        self, frame, ball_analytics, homography, positions: dict, team_name=str, team_color=None
    ) -> None:
        """team_name(team) -> the label shown on the panel ("white", "yellow");
        team_color(team) -> its BGR kit colour, None until the teams are fitted."""
        if self.show_markers and homography is not None:
            ball = ball_analytics.ball
            if ball is not None:
                point = self._project(homography, ball.x, ball.y, frame.shape)
                if point is not None:
                    seen = ball.coasting_s == 0
                    cv2.circle(frame, point, 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
                    cv2.circle(
                        frame, point, 9, self.BALL_COLOR, 2 if seen else 1, lineType=cv2.LINE_AA
                    )
            holder = ball_analytics.holder
            if holder is not None and holder in positions:
                x, y = positions[holder]
                point = self._project(homography, x, y, frame.shape)
                if point is not None:
                    cv2.ellipse(
                        frame, point, (22, 9), 0, 0, 360, self.BALL_COLOR, 2, lineType=cv2.LINE_AA
                    )
        if self.show_passes:
            self._draw_panel(frame, ball_analytics, team_name, team_color or (lambda _team: None))

    PANEL_WIDTH = 260
    PANEL_LINE_PX = 28

    def _draw_panel(self, frame, ball_analytics, team_name, team_color) -> None:
        """Completed passes for both teams, from the first frame to the last:
        the rows are always there (0 until a pass settles) and the counts
        only ever go up. The breakdown (short/long/lost) stays in the
        terminal summary and passes.json."""
        totals = ball_analytics.passes.team_totals()
        rows = []
        for team in (0, 1):
            color = team_color(team)
            label = f"{team_name(team)} team" if color is not None else f"team {team + 1}"
            rows.append((label, totals.get(team, {}).get("completed", 0), color))
        w = frame.shape[1]
        x0, y0 = w - self.PANEL_MARGIN - self.PANEL_WIDTH, self.PANEL_MARGIN
        height = self.PANEL_LINE_PX * (len(rows) + 1) + 12
        overlay = frame[y0 : y0 + height, x0 : x0 + self.PANEL_WIDTH]
        cv2.addWeighted(np.zeros_like(overlay), 0.55, overlay, 0.45, 0, dst=overlay)
        self._panel_text(frame, "Passes completed", (x0 + 10, y0 + self.PANEL_LINE_PX), 0.6)
        for i, (label, count, color) in enumerate(rows):
            baseline = y0 + self.PANEL_LINE_PX * (i + 2)
            if color is not None:
                cv2.rectangle(frame, (x0 + 10, baseline - 14), (x0 + 24, baseline), color, -1)
            self._panel_text(frame, label, (x0 + 32, baseline), 0.6)
            count_text = str(count)
            (tw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            self._panel_text(frame, count_text, (x0 + self.PANEL_WIDTH - 12 - tw, baseline), 0.7)

    @staticmethod
    def _panel_text(frame, text, origin, scale) -> None:
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _project(homography, x, y, shape):
        point = homography @ np.array([x, y, 1.0])
        if abs(point[2]) < 1e-6:
            return None
        px, py = point[0] / point[2], point[1] / point[2]
        h, w = shape[:2]
        if not (-50 <= px <= w + 50 and -50 <= py <= h + 50):
            return None
        return int(px), int(py)


class ResultTimeline:
    """Places each player's box on the exact frame being displayed, from the
    last few resolved AI results, in video time (frame ids).

    A frame that has AI results on both sides -- which PlaybackDelay arranges
    by holding frames back -- gets its boxes interpolated between them, so the
    marker sits on the player in that very frame. A frame newer than every
    result (no delay, or the worker fell behind) gets them extrapolated from
    the two newest results, clamped, so markers keep moving between inference
    cycles instead of freezing -- what MotionExtrapolator did from wall-clock
    time, which was only right while playback was paced.
    """

    HISTORY = 16  # results kept: ~0.3-0.6s at 25-50 results/s, well past any sane delay

    def __init__(self, max_shift_px=120):
        self.max_shift_px = max_shift_px
        self.results = deque(maxlen=self.HISTORY)
        self.synced_frames = 0
        self.extrapolated_frames = 0

    def add(self, result: "InferenceResult") -> None:  # noqa: F821 -- pipeline.InferenceResult; string to avoid a circular import (pipeline.py imports this module)
        if result.frame_id is None:
            return
        if self.results and result.frame_id <= self.results[-1].frame_id:
            return  # already added (the worker has no newer result yet)
        self.results.append(result)

    def boxes_at(self, frame_id: int):
        """(boxes, coasting_progress, synced) for `frame_id`; synced is False
        when the boxes were extrapolated ahead of every result."""
        if not self.results:
            return [], {}, False
        before = next((r for r in reversed(self.results) if r.frame_id <= frame_id), None)
        after = next((r for r in self.results if r.frame_id >= frame_id), None)
        if after is not None:
            self.synced_frames += 1
            if before is None or before is after:
                return after.boxes, after.coasting_progress, True
            t = (frame_id - before.frame_id) / (after.frame_id - before.frame_id)
            return self._interpolate(before, after, t), after.coasting_progress, True

        self.extrapolated_frames += 1
        latest = self.results[-1]
        if len(self.results) < 2:
            return latest.boxes, latest.coasting_progress, False
        extrapolated = self._extrapolate(self.results[-2], latest, frame_id)
        return extrapolated, latest.coasting_progress, False

    def summary(self) -> str:
        total = self.synced_frames + self.extrapolated_frames
        if not total:
            return "Markers: no frames displayed with AI results"
        return (
            f"Markers on their own frame's detections (interpolated): "
            f"{self.synced_frames}/{total}; extrapolated ahead of the newest result: "
            f"{self.extrapolated_frames}/{total}"
        )

    @staticmethod
    def _interpolate(before, after, t: float):
        before_by_id = {box[4]: box for box in before.boxes}
        boxes = []
        for box in after.boxes:
            start = before_by_id.get(box[4])
            if start is None:
                boxes.append(box)  # appeared since `before`: nothing to ease from
                continue
            boxes.append(
                (
                    *(int(round(s + (e - s) * t)) for s, e in zip(start[:4], box[:4], strict=True)),
                    box[4],
                )
            )
        return boxes

    def _extrapolate(self, previous, latest, frame_id: int):
        frames = latest.frame_id - previous.frame_id
        ahead = frame_id - latest.frame_id
        previous_by_id = {box[4]: box for box in previous.boxes}
        boxes = []
        for x1, y1, x2, y2, track_id in latest.boxes:
            start = previous_by_id.get(track_id)
            if start is None:
                boxes.append((x1, y1, x2, y2, track_id))
                continue
            shift_x = self._clamp((x1 - start[0]) / frames * ahead)
            shift_y = self._clamp((y1 - start[1]) / frames * ahead)
            boxes.append((x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y, track_id))
        return boxes

    def _clamp(self, shift: float) -> int:
        return int(max(-self.max_shift_px, min(self.max_shift_px, shift)))


class PlaybackDelay:
    """Holds each frame back a fixed number of frames before it is drawn and
    shown, so by then the AI result for that frame (or one each side of it)
    has usually arrived and ResultTimeline can put markers exactly where the
    players are in it, instead of trailing them by an inference cycle. The
    whole picture is late by the delay; nothing is dropped, and playback
    still paces at the source rate. delay_frames=0 shows every frame as read.
    """

    def __init__(self, delay_frames: int):
        self.delay_frames = max(0, delay_frames)
        self.frames = deque()  # (frame, frame_id), oldest first

    def push(self, frame, frame_id: int) -> None:
        self.frames.append((frame, frame_id))

    def pop(self, flush: bool = False):
        """The next (frame, frame_id) to show, or None while still filling;
        flush=True (the source has ended) empties the buffer regardless."""
        if self.frames and (flush or len(self.frames) > self.delay_frames):
            return self.frames.popleft()
        return None


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

    def smooth(self, boxes, ease: bool = True):
        """ease=False snaps every box to its target (still tracking it for the
        next frame): boxes interpolated onto their own frame are already
        exact, and easing them would only add back a frame of lag."""
        current_ids = set()
        smoothed = []
        for x1, y1, x2, y2, track_id in boxes:
            if track_id < 0:
                smoothed.append((x1, y1, x2, y2, track_id))
                continue

            current_ids.add(track_id)
            target = (float(x1), float(y1), float(x2), float(y2))
            previous = self.positions.get(track_id)
            if not ease or previous is None or self._jumped(previous, target):
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
    worker plus resolving its latest result into identified players, drawing
    markers, cv2.imshow, and the pacer's wait) so we can see exactly where
    main-thread time goes, rather than assuming it's all in one place.
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
            stat("submit + resolve identities (ms)", self.submit_ms),
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


class RawVideoPipe:
    """Hands annotated frames to an ffmpeg-family child process as raw BGR
    video on its stdin. A writer thread does the piping, so the display
    thread only enqueues; the child does the expensive part in its own
    process -- encoding, or putting pixels on screen.
    """

    QUEUE_FRAMES = 8

    def __init__(self, command: list[str]):
        if shutil.which(command[0]) is None:
            sys.exit(f"{command[0]} not found on PATH (brew install ffmpeg)")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.frames = queue.Queue(maxsize=self.QUEUE_FRAMES)
        self.closed = False  # the child exited (e.g. its window was closed)
        self.thread = threading.Thread(target=self._pipe, daemon=True)
        self.thread.start()

    @staticmethod
    def raw_input_args(width: int, height: int, fps: float) -> list[str]:
        return [
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", f"{fps:g}",
            "-i", "-",
        ]  # fmt: skip

    def write(self, frame: np.ndarray) -> None:
        """Blocks only if the child falls QUEUE_FRAMES behind, so a file is
        never missing frames; the caller must not draw on `frame` afterwards."""
        self.frames.put(frame)

    def release(self) -> None:
        self.frames.put(None)
        self.thread.join()
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.wait()

    def _pipe(self) -> None:
        while (frame := self.frames.get()) is not None:
            if self.closed:
                continue  # keep draining so write() never blocks on a dead child
            try:
                self.process.stdin.write(memoryview(np.ascontiguousarray(frame)))
            except BrokenPipeError:
                self.closed = True


class FfmpegOutput(RawVideoPipe):
    """Encodes annotated frames with Apple's hardware H.264 encoder, to a
    file or a live stream URL (rtmp://, srt://, udp://, rtsp://).
    cv2.VideoWriter's mp4v took 10.6ms per 4K frame on the display thread;
    here the encode runs on the media engine, in ffmpeg's process.

    `audio_source` (the input video file) has its audio track, if any, muxed
    back in: every source frame is written, at the source fps, so the two
    line up without resampling. -shortest trims the audio when playback is
    quit early.
    """

    STREAM_FORMATS = {
        "rtmp": "flv",
        "rtmps": "flv",
        "srt": "mpegts",
        "udp": "mpegts",
        "rtsp": "rtsp",
    }
    BITS_PER_PIXEL = 0.1  # ~10 Mbit/s at 1080p50

    def __init__(
        self, target: str, width: int, height: int, fps: float, audio_source: str | None = None
    ):
        bitrate = int(width * height * fps * self.BITS_PER_PIXEL)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *self.raw_input_args(width, height, fps),
        ]  # fmt: skip
        if audio_source is not None:
            command += [
                "-i", audio_source,
                "-map", "0:v", "-map", "1:a?",  # "?": a silent source is fine
                "-c:a", "aac", "-shortest",
            ]  # fmt: skip
        command += [
            "-c:v", "h264_videotoolbox", "-b:v", str(bitrate), "-pix_fmt", "yuv420p",
            "-g", str(round(fps * 2)),  # a keyframe every 2s, so stream viewers can join
        ]  # fmt: skip
        scheme = target.split("://", 1)[0].lower() if "://" in target else None
        if scheme in self.STREAM_FORMATS:
            command += ["-f", self.STREAM_FORMATS[scheme]]
        super().__init__([*command, target])


class FfplayViewer(RawVideoPipe):
    """Shows annotated frames in an ffplay window instead of cv2.imshow. On
    macOS, cv2.waitKey costs ~14-18ms per call whatever the frame size (540p
    to 1080p; 30ms at 4K) -- it waits on the window's repaint -- which caps
    the display loop near 40fps on a 50fps source. ffplay draws on the GPU
    in its own process, so the loop only enqueues the frame and paces with a
    plain sleep. Closing the window (or q in it) ends playback.
    """

    def __init__(self, title: str, width: int, height: int, fps: float):
        super().__init__(
            [
                "ffplay",
                "-hide_banner",
                "-loglevel",
                "error",
                "-window_title",
                title,
                "-autoexit",  # else it idles on after the last frame
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                *self.raw_input_args(width, height, fps),
            ]  # fmt: skip
        )
