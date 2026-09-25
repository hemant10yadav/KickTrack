"""Everything "where is the ball and who has it": the ball track, possession,
and the passes that fall out of possession changing hands (Plan 3.2).

The detector already sees the ball in about half of all frames, as a 10-17px
blob at low confidence, mixed with look-alikes (heads, socks, line crossings).
`BallTracker` turns those candidates into one physically plausible track in
pitch metres; `PossessionTracker` decides who has it; `PassCounter` counts
possession moving from one player to another. All in video time, from the
same gated pitch positions the heatmaps use.
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class BallState:
    t: float
    x: float
    y: float
    vx: float
    vy: float
    coasting_s: float  # seconds since the last accepted sighting (0 = seen now)
    conf: float  # confidence of the last accepted sighting


class BallTracker:
    """One ball track from noisy candidates, in pitch metres.

    Position follows the accepted sighting (lightly smoothed); velocity is the
    displacement over the last VELOCITY_WINDOW_S of sightings and is used only
    to coast through gaps, damped, because a rolling ball slows down. A
    Kalman filter was tried first and dropped: with a gate on top of it, the
    stale velocity after a ball stops at a foot kept dragging the estimate
    away from the very sightings that said it had stopped.

    Two things keep look-alikes out. A candidate is only considered inside a
    gate that grows with time since the last sighting at the ball's top
    speed. And a track that has sat still for STATIC_WINDOW_S with nobody
    within STATIC_PLAYER_M is a fixed feature -- a penalty spot, a line
    crossing, a touchline object (the densest candidate cells on match_4 were
    exactly those; one held the track for seconds until a player walked past
    and "received" it) -- never the ball in play. It is dropped and its spot
    banned for BAN_S. Acquisition needs two consistent sightings, preferring
    one at a player's feet.
    """

    MAX_BALL_SPEED_MS = 35.0
    GATE_BASE_M = 2.0
    GATE_MAX_M = 40.0
    MAX_COAST_S = 1.5
    ACQUIRE_WITHIN_M = 3.0
    VELOCITY_WINDOW_S = 0.12
    COAST_DAMPING_PER_S = 0.3  # velocity kept after 1 s of coasting
    SMOOTHING = 0.6  # weight of the new sighting in the position
    STATIC_WINDOW_S = 2.0
    STATIC_MOVE_M = 1.5  # a track that moved less than this over the window sat still
    STATIC_PLAYER_M = 2.5  # ...with no player this close: a fixed feature
    BAN_RADIUS_M = 2.0
    BAN_S = 20.0
    NEAR_PLAYER_M = 3.0

    def __init__(self):
        self.state: BallState | None = None
        self._hits: list = []  # (t, x, y) accepted sightings, recent
        self._tentative = None
        self._banned: list = []  # (x, y, until_t) spots that behaved like fixed features
        self.sightings = 0
        self.rejected = 0
        self.static_rejected = 0

    def update(self, t: float, candidates: list, positions: dict | None = None) -> BallState | None:
        """candidates: (x_m, y_m, conf) on the pitch; positions: player_id ->
        (x_m, y_m) this result. Returns the ball state (coasting or seen) or
        None when the ball is lost."""
        positions = positions or {}
        self._banned = [b for b in self._banned if b[2] > t]
        before = len(candidates)
        candidates = [c for c in candidates if not self._is_banned(c)]
        self.static_rejected += before - len(candidates)
        if self.state is not None and self._sat_still(t, positions):
            self._banned.append((self.state.x, self.state.y, t + self.BAN_S))
            self.state, self._hits = None, []
            candidates = [c for c in candidates if not self._is_banned(c)]
        if self.state is not None:
            dt = t - self.state.t
            coasting_s = self.state.coasting_s + dt
            gate = min(self.GATE_MAX_M, self.GATE_BASE_M + self.MAX_BALL_SPEED_MS * coasting_s)
            px, py = self._coast(dt)
            hit = self._pick(candidates, px, py, gate)
            self.rejected += len(candidates) - (hit is not None)
            if hit is not None:
                x = self.state.x + self.SMOOTHING * (hit[0] - self.state.x)
                y = self.state.y + self.SMOOTHING * (hit[1] - self.state.y)
                if self.state.coasting_s > 0:
                    x, y = hit[0], hit[1]  # after a gap the old position is stale
                self._hits.append((t, x, y))
                self._hits = [h for h in self._hits if t - h[0] <= self.STATIC_WINDOW_S]
                vx, vy = self._velocity()
                self.sightings += 1
                self.state = BallState(t, x, y, vx, vy, 0.0, hit[2])
                return self.state
            if coasting_s > self.MAX_COAST_S:
                self.state = None
                self._hits = []
                return None
            damp = self.COAST_DAMPING_PER_S**dt
            self.state = BallState(
                t, px, py, self.state.vx * damp, self.state.vy * damp, coasting_s, self.state.conf
            )
            return self.state
        return self._acquire(t, candidates, positions)

    def _coast(self, dt: float) -> tuple[float, float]:
        return self.state.x + self.state.vx * dt, self.state.y + self.state.vy * dt

    def _velocity(self) -> tuple[float, float]:
        if len(self._hits) < 2:
            return 0.0, 0.0
        t1, x1, y1 = self._hits[-1]
        older = [h for h in self._hits if t1 - h[0] >= self.VELOCITY_WINDOW_S] or self._hits[:1]
        t0, x0, y0 = older[-1]
        if t1 <= t0:
            return 0.0, 0.0
        return (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)

    def _acquire(self, t: float, candidates: list, positions: dict) -> BallState | None:
        if not candidates:
            return None
        best = max(candidates, key=lambda c: (self._near_player(c, positions), c[2]))
        if self._tentative is not None:
            t0, x0, y0, _ = self._tentative
            dt = t - t0
            near = [
                c for c in candidates if np.hypot(c[0] - x0, c[1] - y0) <= self.ACQUIRE_WITHIN_M
            ]
            if near and 0 < dt <= self.MAX_COAST_S:
                x, y, conf = max(near, key=lambda c: c[2])
                self._tentative = None
                self._hits = [(t0, x0, y0), (t, x, y)]
                self.sightings += 1
                self.state = BallState(t, x, y, (x - x0) / dt, (y - y0) / dt, 0.0, conf)
                return self.state
        self._tentative = (t, best[0], best[1], best[2])
        return None

    def _near_player(self, candidate, positions: dict) -> bool:
        return any(
            np.hypot(candidate[0] - x, candidate[1] - y) <= self.NEAR_PLAYER_M
            for x, y in positions.values()
        )

    def _pick(self, candidates: list, px: float, py: float, gate: float):
        best, best_score = None, None
        for x, y, conf in candidates:
            d = float(np.hypot(x - px, y - py))
            if d > gate:
                continue
            score = d / gate - 0.5 * conf  # nearest wins; confidence breaks near-ties
            if best_score is None or score < best_score:
                best, best_score = (x, y, conf), score
        return best

    def _is_banned(self, candidate) -> bool:
        return any(
            np.hypot(candidate[0] - x, candidate[1] - y) <= self.BAN_RADIUS_M
            for x, y, _ in self._banned
        )

    def _sat_still(self, t: float, positions: dict) -> bool:
        """The whole recent track within STATIC_MOVE_M and nobody near it."""
        if (
            self.state.coasting_s > 0
            or not self._hits
            or t - self._hits[0][0] < self.STATIC_WINDOW_S
        ):
            return False
        pts = np.array([[x, y] for _, x, y in self._hits])
        if np.ptp(pts, axis=0).max() > self.STATIC_MOVE_M:
            return False
        return not any(
            np.hypot(self.state.x - x, self.state.y - y) <= self.STATIC_PLAYER_M
            for x, y in positions.values()
        )


@dataclass
class Possession:
    player_id: int
    team: int | None
    start_t: float
    end_t: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


class PossessionTracker:
    """Who has the ball: the nearest player within CONTROL_M of a tracked (not
    long-coasting) ball for HOLD_RESULTS results in a row, until the ball is
    RELEASE_M away for RELEASE_RESULTS results or lost. Distances are between
    the ball and the player's feet on the pitch plane, so a ball in the air
    passing over a player briefly reads as close; the hold requirement is what
    keeps that from counting."""

    CONTROL_M = 1.5
    RELEASE_M = 2.5
    HOLD_RESULTS = 3
    RELEASE_RESULTS = 3
    MAX_COAST_FOR_CONTROL_S = 0.3

    def __init__(self):
        self.current: Possession | None = None
        self.history: list[Possession] = []
        self._candidate = None  # (player_id, consecutive results)
        self._away = 0

    def update(
        self, t: float, ball: BallState | None, positions: dict, team_of
    ) -> Possession | None:
        """Returns a possession that *ended* on this update, if any."""
        if ball is None or ball.coasting_s > self.MAX_COAST_FOR_CONTROL_S or not positions:
            return self._end(t) if ball is None else self._maybe_release(t, None)
        nearest, dist = None, None
        for pid, (x, y) in positions.items():
            d = float(np.hypot(x - ball.x, y - ball.y))
            if dist is None or d < dist:
                nearest, dist = pid, d
        if self.current is not None:
            holder_xy = positions.get(self.current.player_id)
            holder_d = (
                None
                if holder_xy is None
                else float(np.hypot(holder_xy[0] - ball.x, holder_xy[1] - ball.y))
            )
            if holder_d is not None and holder_d <= self.RELEASE_M:
                self._away = 0
                self.current.end_t, self.current.end_xy = t, holder_xy
                return None
            ended = self._maybe_release(
                t, nearest if dist is not None and dist <= self.CONTROL_M else None
            )
            if ended is not None or self.current is not None:
                return ended
        if dist is not None and dist <= self.CONTROL_M:
            if self._candidate is not None and self._candidate[0] == nearest:
                self._candidate = (nearest, self._candidate[1] + 1)
            else:
                self._candidate = (nearest, 1)
            if self._candidate[1] >= self.HOLD_RESULTS:
                xy = positions[nearest]
                self.current = Possession(nearest, team_of(nearest), t, t, xy, xy)
                self._candidate = None
                self._away = 0
        else:
            self._candidate = None
        return None

    def _maybe_release(self, t: float, other_holder) -> Possession | None:
        if self.current is None:
            return None
        self._away += 1
        if self._away >= self.RELEASE_RESULTS or other_holder is not None:
            return self._end(t)
        return None

    def _end(self, t: float) -> Possession | None:
        if self.current is None:
            return None
        ended, self.current = self.current, None
        self._away = 0
        self.history.append(ended)
        return ended


@dataclass
class PassEvent:
    t: float
    from_player: int
    to_player: int
    team: int | None
    completed: bool  # receiver on the same team
    distance_m: float


class PassCounter:
    """A pass is possession moving from one player to a different one within
    MAX_GAP_S. Completed when the receiver is a teammate, lost when an
    opponent gets it. Short vs long is the pitch distance the ball travelled,
    split at LONG_PASS_M (Opta's long-ball line is ~30 m)."""

    MAX_GAP_S = 4.0
    LONG_PASS_M = 30.0

    def __init__(self):
        self.events: list[PassEvent] = []
        self._last_ended: Possession | None = None

    def on_possession_ended(self, ended: Possession) -> None:
        self._last_ended = ended

    def on_possession_started(self, started: Possession) -> None:
        prev = self._last_ended
        if prev is None or prev.player_id == started.player_id:
            return
        if started.start_t - prev.end_t > self.MAX_GAP_S:
            return
        distance = float(
            np.hypot(started.start_xy[0] - prev.end_xy[0], started.start_xy[1] - prev.end_xy[1])
        )
        completed = prev.team is not None and prev.team == started.team
        self.events.append(
            PassEvent(
                started.start_t, prev.player_id, started.player_id, prev.team, completed, distance
            )
        )
        self._last_ended = None

    def team_totals(self) -> dict:
        totals = {}
        for e in self.events:
            row = totals.setdefault(e.team, {"completed": 0, "lost": 0, "short": 0, "long": 0})
            row["completed" if e.completed else "lost"] += 1
            if e.completed:
                row["long" if e.distance_m >= self.LONG_PASS_M else "short"] += 1
        return totals


class BallAnalytics:
    """Per-result driver: candidates -> ball track -> possession -> passes."""

    def __init__(self, fps: float):
        self.fps = fps
        self.tracker = BallTracker()
        self.possession = PossessionTracker()
        self.passes = PassCounter()
        self.ball: BallState | None = None
        self.results = 0
        self.results_with_ball = 0
        self._last_frame_id = None
        self._last_holder = None

    def record(self, frame_id: int, candidates_pitch: list, positions: dict, team_of) -> None:
        if frame_id is None or frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        t = frame_id / self.fps
        self.results += 1
        self.ball = self.tracker.update(t, candidates_pitch, positions)
        self.results_with_ball += self.ball is not None
        ended = self.possession.update(t, self.ball, positions, team_of)
        if ended is not None:
            self.passes.on_possession_ended(ended)
        current = self.possession.current
        holder = None if current is None else current.player_id
        if holder is not None and holder != self._last_holder:
            self.passes.on_possession_started(current)
        self._last_holder = holder

    @property
    def holder(self) -> int | None:
        return None if self.possession.current is None else self.possession.current.player_id

    def summary(self, team_name=str) -> str:
        """team_name(team) -> label, e.g. TeamClassifier.team_name."""
        lines = [
            f"Ball tracked in {self.results_with_ball}/{self.results} results "
            f"({self.tracker.sightings} sightings, {self.tracker.rejected} candidates rejected, "
            f"{self.tracker.static_rejected} as fixed features); "
            f"possessions: {len(self.possession.history) + (self.possession.current is not None)}",
        ]
        for team, row in sorted(self.passes.team_totals().items(), key=lambda kv: str(kv[0])):
            lines.append(
                f"  {team_name(team)}: {row['completed']} passes "
                f"({row['short']} short, {row['long']} long), {row['lost']} lost"
            )
        return "\n".join(lines)

    def write(self, out_dir, team_name=str) -> None:
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "passes.json").write_text(json.dumps(self.as_dict(team_name), indent=2))

    def as_dict(self, team_name=str) -> dict:
        return {
            "results": self.results,
            "results_with_ball": self.results_with_ball,
            "teams": {team_name(k): v for k, v in self.passes.team_totals().items()},
            "passes": [
                {
                    "t": round(e.t, 2),
                    "from": e.from_player,
                    "to": e.to_player,
                    "team": team_name(e.team),
                    "completed": e.completed,
                    "distance_m": round(e.distance_m, 1),
                }
                for e in self.passes.events
            ],
        }
