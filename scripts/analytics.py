"""Everything "what do we measure from where players were": projecting tracked
boxes onto the pitch, per-player distance/speed, and heatmaps (Plan 3).

Positions come from IdentityResolver's confirmed boxes (one sample per worker
result, in *video* time) and the pitch homography from calibration. Every
sample is gated -- no homography, an occluded (contaminated) box, or a point
that lands off the pitch is dropped rather than fed into a number that would
then be silently wrong. The camera follows play, so a player is only measured
while in frame: `tracked_s` is reported next to `distance_m` for that reason.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from scripts.calibration import PITCH_LINES

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PITCH_MARGIN_M = (
    3.0  # a foot position this far outside the lines is still a player on the pitch edge
)


class PitchProjector:
    """Image pixel -> pitch metres (centred at the centre spot, x along the
    length, y along the width) through the inverse of the calibration
    homography, which maps pitch -> image (see scripts/calibration.py)."""

    def __init__(self, homography: np.ndarray | None):
        self.inverse = None
        if homography is not None and abs(np.linalg.det(homography)) > 1e-9:
            self.inverse = np.linalg.inv(homography)

    @property
    def available(self) -> bool:
        return self.inverse is not None

    def to_pitch(self, px: float, py: float) -> tuple[float, float] | None:
        """None when there is no calibration or the point is not on the pitch."""
        if self.inverse is None:
            return None
        point = self.inverse @ np.array([px, py, 1.0])
        if abs(point[2]) < 1e-9:
            return None
        x, y = float(point[0] / point[2]), float(point[1] / point[2])
        if (
            abs(x) > PITCH_LENGTH_M / 2 + PITCH_MARGIN_M
            or abs(y) > PITCH_WIDTH_M / 2 + PITCH_MARGIN_M
        ):
            return None
        return x, y


@dataclass
class PlayerTrace:
    """One player's measured presence on the pitch.

    Distance is not the sum of frame-to-frame steps: a box's bottom edge
    jitters by a pixel or two every frame (a few centimetres at this camera
    scale) and swings with the running stride, and at 50fps summing that
    would invent metres per second of motion for a player standing still.
    Positions are averaged over WINDOW_S buckets first and distance is the
    path length between bucket means; a bucket is only chained to the
    previous one when they are within MAX_GAP_S, so time out of frame (the
    camera panned away) adds nothing. Measured on match_5 (docs/PLAN.md Plan
    3.1): 0.2s and 0.5s windows agree within 6%, so the motion that survives
    0.5s is real, not noise. Speed is the same bucket displacement over its
    time; top speed is taken over two consecutive buckets (~1s) so a single
    noisy bucket cannot set it. Below MIN_SPEED_MS (a third of walking pace)
    the residual jitter of a standing player still drifts a few centimetres
    per bucket, so such steps count as standing; above MAX_SPEED_MS it is a
    calibration jump or an identity error, not a footballer. Both are dropped
    from distance and speed.
    """

    player_id: int
    team: int | None = None
    samples: int = 0
    tracked_s: float = 0.0
    distance_m: float = 0.0
    top_speed_ms: float = 0.0
    heat: np.ndarray = field(
        default_factory=lambda: np.zeros((int(PITCH_WIDTH_M), int(PITCH_LENGTH_M)))
    )
    _bucket: list = field(default_factory=list)  # (t, x, y) samples of the open bucket
    _last_mean: tuple | None = None  # (t, x, y) of the last closed bucket
    _last_step: tuple | None = None  # (metres, seconds) of the last counted step
    _last_t: float | None = None
    _pending_shift: tuple = (0.0, 0.0)  # calibration shift to cancel from the next step

    WINDOW_S = 0.5
    MAX_GAP_S = 1.0
    MIN_SPEED_MS = 0.5
    MAX_SPEED_MS = 12.0

    def add(self, t: float, x: float, y: float, frame_dt: float) -> None:
        """frame_dt: the video time one sample stands for when this player was
        not seen in the previous result (the first sample, or one after time
        out of frame); otherwise the real gap since their previous sample."""
        seen_recently = self._last_t is not None and 0 < t - self._last_t <= self.MAX_GAP_S
        dt = t - self._last_t if seen_recently else frame_dt
        self._last_t = t
        self.samples += 1
        self.tracked_s += dt
        col = int(np.clip(x + PITCH_LENGTH_M / 2, 0, PITCH_LENGTH_M - 1e-6))
        row = int(np.clip(y + PITCH_WIDTH_M / 2, 0, PITCH_WIDTH_M - 1e-6))
        self.heat[row, col] += dt
        if self._bucket and t - self._bucket[0][0] >= self.WINDOW_S:
            self._close_bucket()
        self._bucket.append((t, x, y))

    def _close_bucket(self) -> None:
        arr = np.array(self._bucket)
        mean = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
        self._bucket = []
        counted = None
        if self._last_mean is not None:
            dt = mean[0] - self._last_mean[0]
            if 0 < dt <= self.MAX_GAP_S:
                dx = mean[1] - self._last_mean[1] - self._pending_shift[0]
                dy = mean[2] - self._last_mean[2] - self._pending_shift[1]
                step = float(np.hypot(dx, dy))
                speed = step / dt
                if self.MIN_SPEED_MS <= speed <= self.MAX_SPEED_MS:
                    self.distance_m += step
                    counted = (step, dt)
                    if self._last_step is not None:
                        prev_step, prev_dt = self._last_step
                        self.top_speed_ms = max(
                            self.top_speed_ms, (step + prev_step) / (dt + prev_dt)
                        )
        self._last_mean = mean
        self._last_step = counted
        self._pending_shift = (0.0, 0.0)

    def note_calibration_shift(self, dx: float, dy: float) -> None:
        """The pitch mapping just moved everyone by (dx, dy). The open bucket
        is closed on the pre-shift positions, and the shift is taken out of
        the one step that straddles it, so only motion relative to the pitch
        survives. Positions themselves (heatmap) are left as measured."""
        if self._bucket:
            self._close_bucket()
        self._pending_shift = (self._pending_shift[0] + dx, self._pending_shift[1] + dy)

    def finish(self) -> None:
        if self._bucket:
            self._close_bucket()

    def as_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "samples": self.samples,
            "tracked_s": round(self.tracked_s, 2),
            "distance_m": round(self.distance_m, 1),
            "top_speed_ms": round(self.top_speed_ms, 2),
        }


class MatchAnalytics:
    """Accumulates per-player traces from each resolved worker result.

    One correction is global: when the median shift of the players on the
    pitch between two consecutive results is JUMP_M or more, the calibration
    moved, not the players (no team averages 10 m/s in one direction). Measured
    on match_5 (docs/PLAN.md Plan 3.1) this happens ~4% of results, in pairs
    25 frames apart at every keyframe hand-over -- the new homography arrives
    ~0.5s late, the propagator resets to that stale pose, then catches up --
    plus a few metre-sized snaps on fast pans. The median shift is cancelled
    out of every player's step across that moment (see
    PlayerTrace.note_calibration_shift); it is counted, not dropped, so the
    time and the motion on either side are kept.
    """

    JUMP_M = 0.2
    JUMP_MIN_PLAYERS = 4

    def __init__(self, fps: float):
        self.fps = fps
        self.players: dict[int, PlayerTrace] = {}
        self.samples_dropped = {"no_homography": 0, "occluded": 0, "off_pitch": 0}
        self.calibration_shifts = 0  # results where the mapping, not the players, moved
        self.latest_positions: dict[int, tuple[float, float]] = {}  # pitch metres, last result
        self._last_frame_id = None
        self._last_positions: dict[int, tuple[float, float]] = {}

    def record(self, frame_id: int, boxes: list, occluded_ids: set, homography, team_of) -> None:
        """boxes: (x1, y1, x2, y2, player_id) confirmed boxes of one worker
        result; team_of(player_id) -> team or None."""
        if frame_id is None or frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        t = frame_id / self.fps
        frame_dt = 1 / self.fps
        projector = PitchProjector(homography)
        positions = {}
        for x1, _y1, x2, y2, player_id in boxes:
            if player_id < 0:
                continue
            if not projector.available:
                self.samples_dropped["no_homography"] += 1
                continue
            if player_id in occluded_ids:
                self.samples_dropped["occluded"] += 1
                continue
            position = projector.to_pitch((x1 + x2) / 2, y2)  # feet, not box centre
            if position is None:
                self.samples_dropped["off_pitch"] += 1
                continue
            positions[player_id] = position

        shift = self._calibration_shift(positions)
        if shift is not None:
            self.calibration_shifts += 1
            for trace in self.players.values():
                trace.note_calibration_shift(*shift)
        self._last_positions = positions
        self.latest_positions = positions

        for player_id, (x, y) in positions.items():
            trace = self.players.setdefault(player_id, PlayerTrace(player_id))
            team = team_of(player_id)
            if team is not None:
                trace.team = team
            trace.add(t, x, y, frame_dt)

    def _calibration_shift(self, positions: dict) -> tuple[float, float] | None:
        """The (dx, dy) everyone moved by since the previous result, when that
        is a calibration jump; None when it is just play."""
        common = [pid for pid in positions if pid in self._last_positions]
        if len(common) < self.JUMP_MIN_PLAYERS:
            return None
        deltas = np.array(
            [
                [
                    positions[pid][0] - self._last_positions[pid][0],
                    positions[pid][1] - self._last_positions[pid][1],
                ]
                for pid in common
            ]
        )
        median = np.median(deltas, axis=0)
        if float(np.hypot(*median)) < self.JUMP_M:
            return None
        return float(median[0]), float(median[1])

    def distance_of(self, player_id: int) -> float | None:
        trace = self.players.get(player_id)
        return None if trace is None else trace.distance_m

    def finish(self) -> None:
        for trace in self.players.values():
            trace.finish()

    def team_heat(self, team: int) -> np.ndarray:
        heat = np.zeros((int(PITCH_WIDTH_M), int(PITCH_LENGTH_M)))
        for trace in self.players.values():
            if trace.team == team:
                heat += trace.heat
        return heat

    def summary(self, top: int = 12) -> str:
        traces = sorted(self.players.values(), key=lambda p: -p.distance_m)
        lines = [
            f"Analytics samples dropped: {self.samples_dropped}; "
            f"calibration shifts cancelled: {self.calibration_shifts}",
            f"{'player':>6} {'team':>4} {'tracked_s':>9} {'distance_m':>10} {'top_speed':>9}",
        ]
        for p in traces[:top]:
            lines.append(
                f"{p.player_id:>6} {str(p.team):>4} {p.tracked_s:>9.1f} {p.distance_m:>10.1f} "
                f"{p.top_speed_ms:>7.1f}m/s"
            )
        if len(traces) > top:
            lines.append(f"  ... and {len(traces) - top} more players")
        return "\n".join(lines)

    def write(self, out_dir: str | Path, team_color=None, min_tracked_s: float = 2.0) -> None:
        """stats.json plus a heatmap PNG per player (tracked at least
        min_tracked_s) and per team."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.finish()
        stats = [p.as_dict() for p in sorted(self.players.values(), key=lambda p: -p.distance_m)]
        (out / "stats.json").write_text(
            json.dumps(
                {
                    "fps": self.fps,
                    "samples_dropped": self.samples_dropped,
                    "calibration_shifts": self.calibration_shifts,
                    "players": stats,
                },
                indent=2,
            )
        )
        teams = {p.team for p in self.players.values() if p.team is not None}
        for team in sorted(teams):
            color = team_color(team) if team_color else None
            cv2.imwrite(
                str(out / f"heatmap_team_{team}.png"),
                render_heatmap(self.team_heat(team), f"team {team}", color),
            )
        for p in self.players.values():
            if p.tracked_s < min_tracked_s:
                continue
            color = team_color(p.team) if (team_color and p.team is not None) else None
            title = (
                f"player {p.player_id}  {p.distance_m:.0f}m in {p.tracked_s:.0f}s, "
                f"top {p.top_speed_ms:.1f}m/s"
            )
            cv2.imwrite(
                str(out / f"heatmap_player_{p.player_id}.png"), render_heatmap(p.heat, title, color)
            )


@dataclass
class TeamShape:
    """One team's shape at one result, in pitch metres. A dimension is None
    when the camera cuts the team off on that axis (see TeamShapeAnalytics)."""

    team: int
    players: int
    hull: np.ndarray  # (k, 2) outline of the visible outfield players
    player_ids: tuple  # who the shape was measured from (officials dropped)
    width_m: float | None
    depth_m: float | None
    area_m2: float | None
    line_m: float | None  # defensive line: the last outfield man's distance from his own goal line


class TeamShapeAnalytics:
    """Width, depth, area and defensive line of each team's outfield players
    (teams 0 and 1; keepers and officials are the classifier's OTHER_TEAM and
    are left out), per worker result, split by who has the ball.

    The camera follows play, so a team is often only partly in frame: on
    match_5 both teams' rearmost player sat on the halfway line for 15s
    because that was the edge of the picture, not of the team. A dimension is
    therefore only measured when the camera sees EDGE_M past the team on both
    ends of that axis (checked at several points along the edge, through the
    homography into the frame); past a goal line or touchline nobody can
    stand, so an edge there only needs the line itself in view. Otherwise it
    is None -- "not measurable right now" -- never a smaller number.

    Which goal a team defends is not known up front. Defenders stand
    goal-side of the opponents they mark, so over time the team defending the
    +x goal has its centroid further toward +x than the other team's, whether
    it is defending deep or attacking (then the opponents' defence is deeper
    still). Measured on match_5: 6-10 m in every 3s window. The mean gap is
    accumulated over results where both teams are visible and the direction
    is committed once DIRECTION_MIN_S of them agree by DIRECTION_MARGIN_M;
    the defensive line is only measured after that.
    """

    MIN_PLAYERS = 6
    MAX_OUTFIELD = 10
    LINE_ZONE_M = 1.0
    ISOLATED_M = 30.0
    EDGE_M = 5.0
    EDGE_SAMPLES = 3
    DIRECTION_MIN_S = 2.0
    DIRECTION_MARGIN_M = 2.0
    DISPLAY_HOLD_S = 1.0  # the panel keeps a value this long after it was last measured
    DISPLAY_SMOOTH_S = 0.5
    SERIES_BUCKET_S = 0.5
    # A team's shape changes over seconds, and the panel smooths over
    # DISPLAY_SMOOTH_S anyway; 10 samples a second instead of every result
    # (up to 50) keeps this off the display thread's budget -- on match_5 the
    # per-result version cost the worker 15-60 more skipped frames of 1750.
    SAMPLE_S = 0.1
    # team_of is TeamHistory's 10 s majority in the pipeline, ~1 ms for a
    # frame's ~25 players every result at 50 fps, on the display thread; the
    # answer cannot move in a quarter of a second, so it is asked that often.
    TEAM_REFRESH_S = 0.25
    METRICS = ("width_m", "depth_m", "area_m2", "line_m")

    def __init__(self, fps: float):
        self.fps = fps
        self.latest: dict[int, TeamShape] = {}
        self.own_goal: dict[int, int] | None = None  # team -> sign of the goal it defends
        self._direction_sum = 0.0
        self._direction_results = 0
        self._last_frame_id = None
        self._samples: list[tuple[float, int, str | None, TeamShape]] = []
        self._display: dict[tuple[int, str], tuple[float, float]] = {}  # -> (value, last_t)
        self.results = 0
        self.results_measured = {team: dict.fromkeys(self.METRICS, 0) for team in (0, 1)}
        self.results_visible = {0: 0, 1: 0}
        self._teams: dict[int, tuple] = {}  # player_id -> (team, when it was asked)

    def record(
        self, frame_id, positions: dict, team_of, homography, frame_size, possession_team=None
    ) -> None:
        """positions: player_id -> (x_m, y_m) of one result (MatchAnalytics'
        latest_positions); homography: pitch -> working-frame pixels, the one
        those positions came through; frame_size: (w, h) of that frame;
        possession_team: the team in possession at this result, or None.
        Sampled every SAMPLE_S of video time; results in between are skipped."""
        if frame_id is None or frame_id == self._last_frame_id:
            return
        t = frame_id / self.fps
        since = None if self._last_frame_id is None else t - self._last_frame_id / self.fps
        if since is not None and 0 <= since < self.SAMPLE_S - 1e-9:
            return
        self._last_frame_id = frame_id
        self.results += 1
        members = {0: [], 1: []}
        for player_id in positions:
            team = self._team(player_id, t, team_of)
            if team in members:
                members[team].append(player_id)
        teams = {}
        for team, ids in members.items():
            if not ids:
                continue
            p = np.array([positions[pid] for pid in ids])
            keep = self._outfield(p)
            if len(keep) >= self.MIN_PLAYERS:
                teams[team] = (p[keep], tuple(ids[i] for i in keep))
        if len(teams) == 2:
            self._direction_sum += float(teams[1][0][:, 0].mean() - teams[0][0][:, 0].mean())
            self._direction_results += 1
            self._update_direction()
        self.latest = {}
        for team, (p, ids) in teams.items():
            shape = self._measure(team, p, ids, homography, frame_size)
            self.latest[team] = shape
            self.results_visible[team] += 1
            phase = None
            if possession_team in (0, 1):
                phase = "in" if possession_team == team else "out"
            self._samples.append((t, team, phase, shape))
            for metric in self.METRICS:
                value = getattr(shape, metric)
                if value is None:
                    continue
                self.results_measured[team][metric] += 1
                key = (team, metric)
                prev = self._display.get(key)
                if prev is not None and t - prev[1] <= self.DISPLAY_HOLD_S:
                    a = min(1.0, (t - prev[1]) / self.DISPLAY_SMOOTH_S)
                    value = prev[0] + a * (value - prev[0])
                self._display[key] = (value, t)

    def _team(self, player_id: int, t: float, team_of):
        cached = self._teams.get(player_id)
        if cached is None or not 0 <= t - cached[1] < self.TEAM_REFRESH_S:
            cached = (team_of(player_id), t)
            self._teams[player_id] = cached
        return cached[0]

    def _outfield(self, p: np.ndarray) -> np.ndarray:
        """The team's points without the officials and staff the classifier
        put on it. Assistant referees run the touchlines and stewards stand at
        the corners, often in a colour near one kit: match_5's far-side
        linesman read sky blue for 2s and stretched City's width to 68 m;
        match_6 had one linesman on each team at once, match_4 one in
        Watford's yellow. A point on or outside a pitch line (within
        LINE_ZONE_M) with no teammate within ISOLATED_M is dropped. Measured
        over all five clips, the officials caught this way stood 33-44 m from
        the nearest player of "their" team; real players on the line came
        closer -- the nearest call, a Tottenham throw-in taker on match_4, at
        26-30 m -- so a few seconds of official survive at 30 m, where a lower
        threshold would cut real wingers. Then at most MAX_OUTFIELD are
        kept, the most isolated going first. Returns the indices kept.
        """
        keep = np.arange(len(p))
        if len(p) < 2:
            return keep
        nearest = self._nearest_teammate(p)
        on_line = (np.abs(p[:, 1]) >= PITCH_WIDTH_M / 2 - self.LINE_ZONE_M) | (
            np.abs(p[:, 0]) >= PITCH_LENGTH_M / 2 - self.LINE_ZONE_M
        )
        keep = keep[~(on_line & (nearest > self.ISOLATED_M))]
        while len(keep) > self.MAX_OUTFIELD:
            keep = np.delete(keep, int(np.argmax(self._nearest_teammate(p[keep]))))
        return keep

    @staticmethod
    def _nearest_teammate(p: np.ndarray) -> np.ndarray:
        d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)

    def _update_direction(self) -> None:
        if self._direction_results < self.DIRECTION_MIN_S * self._results_per_s():
            return
        mean = self._direction_sum / self._direction_results
        if abs(mean) < self.DIRECTION_MARGIN_M:
            self.own_goal = None
            return
        side = 1 if mean > 0 else -1  # team 1 defends the +x goal when it stands further +x
        self.own_goal = {1: side, 0: -side}

    def _results_per_s(self) -> float:
        if self._last_frame_id is None or self.results < 2:
            return self.fps
        return self.results / (self._last_frame_id / self.fps)

    def _measure(self, team: int, p: np.ndarray, ids: tuple, homography, frame_size) -> TeamShape:
        hull = cv2.convexHull(p.astype(np.float32)).reshape(-1, 2)
        xs, ys = p[:, 0], p[:, 1]
        seen = lambda x, y: self._in_view(homography, frame_size, x, y)  # noqa: E731
        x_along = np.linspace(xs.min(), xs.max(), self.EDGE_SAMPLES)
        y_along = np.linspace(ys.min(), ys.max(), self.EDGE_SAMPLES)
        back_x, front_x = (
            self._beyond(xs.min(), -1, PITCH_LENGTH_M),
            self._beyond(xs.max(), 1, PITCH_LENGTH_M),
        )
        low_y, high_y = (
            self._beyond(ys.min(), -1, PITCH_WIDTH_M),
            self._beyond(ys.max(), 1, PITCH_WIDTH_M),
        )
        closed = {
            -1: all(seen(back_x, y) for y in y_along),  # the -x end
            1: all(seen(front_x, y) for y in y_along),  # the +x end
        }
        width_closed = all(seen(x, low_y) for x in x_along) and all(
            seen(x, high_y) for x in x_along
        )
        depth_closed = closed[-1] and closed[1]
        width = float(np.ptp(ys)) if width_closed else None
        depth = float(np.ptp(xs)) if depth_closed else None
        area = float(cv2.contourArea(hull)) if width_closed and depth_closed else None
        line = None
        if self.own_goal is not None and closed[self.own_goal[team]]:
            goal = self.own_goal[team]
            depth_from_goal = PITCH_LENGTH_M / 2 - goal * xs  # 0 on the own goal line
            line = float(depth_from_goal.min())  # the last man, as the offside line is drawn
        return TeamShape(team, len(p), hull, ids, width, depth, area, line)

    def _beyond(self, edge: float, direction: int, length: float) -> float:
        """The point EDGE_M past a team's edge, stopped at the pitch line."""
        return float(np.clip(edge + direction * self.EDGE_M, -length / 2, length / 2))

    @staticmethod
    def _in_view(homography, frame_size, x: float, y: float) -> bool:
        if homography is None or frame_size is None:
            return False
        point = homography @ np.array([x, y, 1.0])
        if point[2] <= 1e-9:
            return False
        px, py = point[0] / point[2], point[1] / point[2]
        w, h = frame_size
        return 0 <= px < w and 0 <= py < h

    def display_value(self, team: int, metric: str, t: float | None = None) -> float | None:
        """The smoothed value for the live panel: None when it has not been
        measured in the last DISPLAY_HOLD_S."""
        entry = self._display.get((team, metric))
        if entry is None:
            return None
        now = t if t is not None else (self._last_frame_id or 0) / self.fps
        return entry[0] if now - entry[1] <= self.DISPLAY_HOLD_S else None

    def team_stats(self, team: int) -> dict:
        """Median of each metric over the results where it was measured,
        overall and split by possession, and how often it was measurable."""
        out = {"results_visible": self.results_visible[team]}
        for metric in self.METRICS:
            entry = {}
            for label, phase in (
                ("all", ...),
                ("in_possession", "in"),
                ("out_of_possession", "out"),
            ):
                values = [
                    getattr(s, metric)
                    for _t, tm, ph, s in self._samples
                    if tm == team
                    and getattr(s, metric) is not None
                    and (phase is ... or ph == phase)
                ]
                entry[label] = round(float(np.median(values)), 1) if values else None
                entry[f"{label}_results"] = len(values)
            out[metric] = entry
        return out

    def series(self) -> list[dict]:
        """Per SERIES_BUCKET_S of video, each team's median of every metric
        measured in that bucket (None when none was)."""
        buckets: dict[int, list] = {}
        for t, team, phase, shape in self._samples:
            buckets.setdefault(int(t / self.SERIES_BUCKET_S), []).append((team, phase, shape))
        rows = []
        for b in sorted(buckets):
            row = {"t": round(b * self.SERIES_BUCKET_S, 2)}
            for team in (0, 1):
                entries = [(ph, s) for tm, ph, s in buckets[b] if tm == team]
                if not entries:
                    continue
                team_row = {}
                for metric in self.METRICS:
                    values = [
                        getattr(s, metric) for _ph, s in entries if getattr(s, metric) is not None
                    ]
                    team_row[metric] = round(float(np.median(values)), 1) if values else None
                phases = [ph for ph, _s in entries if ph is not None]
                team_row["phase"] = max(set(phases), key=phases.count) if phases else None
                row[str(team)] = team_row
            rows.append(row)
        return rows

    def summary(self, team_name=str) -> str:
        lines = [
            f"Team shape: {self.results} results; defending "
            + (
                ", ".join(
                    f"{team_name(team)} {'+x' if side > 0 else '-x'}"
                    for team, side in sorted(self.own_goal.items())
                )
                if self.own_goal
                else "direction not known"
            ),
            f"{'team':>12} {'metric':>8} {'measured':>9} {'all':>7} {'in poss':>8} {'out poss':>9}",
        ]
        for team in (0, 1):
            stats = self.team_stats(team)
            for metric in self.METRICS:
                m = stats[metric]
                share = (
                    f"{100 * self.results_measured[team][metric] / self.results_visible[team]:.0f}%"
                    if self.results_visible[team]
                    else "-"
                )
                fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
                lines.append(
                    f"{team_name(team):>12} {metric.removesuffix('_m').removesuffix('_m2'):>8} "
                    f"{share:>9} {fmt(m['all']):>7} {fmt(m['in_possession']):>8} "
                    f"{fmt(m['out_of_possession']):>9}"
                )
        return "\n".join(lines)

    def write(self, out_dir: str | Path, team_name=str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "shape.json").write_text(
            json.dumps(
                {
                    "fps": self.fps,
                    "own_goal": None
                    if self.own_goal is None
                    else {team_name(t): side for t, side in self.own_goal.items()},
                    "teams": {team_name(team): self.team_stats(team) for team in (0, 1)},
                    "series": self.series(),
                },
                indent=2,
            )
        )


def render_pitch(scale: int = 8) -> np.ndarray:
    """Green pitch with white markings, `scale` px per metre."""
    w, h = int(PITCH_LENGTH_M * scale), int(PITCH_WIDTH_M * scale)
    img = np.full((h, w, 3), (60, 140, 60), dtype=np.uint8)

    def px(x, y):
        return int((x + PITCH_LENGTH_M / 2) * scale), int((y + PITCH_WIDTH_M / 2) * scale)

    for (x1, y1), (x2, y2) in PITCH_LINES:
        cv2.line(img, px(x1, y1), px(x2, y2), (255, 255, 255), 2)
    cv2.circle(img, px(0, 0), int(9.15 * scale), (255, 255, 255), 2)
    return img


def render_heatmap(heat: np.ndarray, title: str, color=None, scale: int = 8) -> np.ndarray:
    """Time-weighted presence drawn over the pitch. Blurred at ~2m so a
    1m-bin histogram reads as a field, not confetti; colour in the team's
    jersey colour when known."""
    img = render_pitch(scale)
    if heat.sum() <= 0:
        cv2.putText(
            img,
            title + " (no samples)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return img
    big = cv2.resize(
        heat.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    big = cv2.GaussianBlur(big, (0, 0), 2 * scale)
    alpha = np.clip(big / big.max(), 0, 1) ** 0.6
    tint = np.array(color if color is not None else (0, 0, 255), dtype=np.float32)
    img = (img * (1 - 0.85 * alpha[..., None]) + tint * (0.85 * alpha[..., None])).astype(np.uint8)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img
