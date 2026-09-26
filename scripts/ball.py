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

from scripts.analytics import PITCH_LENGTH_M
from scripts.player import OTHER_TEAM


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
    completed: bool  # receiver on the same team, as known at the time
    distance_m: float
    to_team: int | None = None  # the receiver's side at the time (a keeper's, by position)
    from_t: float | None = None  # when the passer last had the ball


class PassCounter:
    """A pass is possession moving from one player to a different one within
    MAX_GAP_S. Completed when the receiver is a teammate, lost when an
    opponent gets it. Short vs long is the pitch distance the ball travelled,
    split at LONG_PASS_M (Opta's long-ball line is ~30 m).

    Possessions with team OTHER_TEAM (a referee, or a keeper whose side could
    not be told) are not part of the chain: the ball brushing past the referee
    neither ends a move nor starts one. A possession further from the last one
    than the ball can travel in the time between them is the track having
    jumped to a look-alike, and is skipped the same way; so is a pass whose
    receiver had the ball for under BRIEF_S after it came in faster than
    BRIEF_MAX_SPEED_MS -- the track hopping across look-alikes (three 20 m+
    jumps in 1.3 s on one match_5 run, each ending in a 0-0.06 s
    "possession" reached at 30+ m/s), where real first-time touches came in
    at 7-15 m/s.

    An opponent's brief touch (under BRIEF_S) between two teammates leaves the
    ball where it was going -- a pass through a City player's legs on
    match_5 (20.2 s) read as white -> City -> white -- so the pass is booked
    from the first teammate to the second, completed, as a deflected pass
    that still reaches a teammate is.

    Events are counted, and their teams frozen, only SETTLE_S after they
    happen, from each player's team reads around the pass (TeamHistory): the
    reads just after a pass are what tell a receiver who had only just
    appeared, and a total that only ever goes up must not count a pass for the
    wrong side first. Until the classifier has fitted, nothing settles; those
    passes are counted as soon as it has."""

    MAX_GAP_S = 4.0
    LONG_PASS_M = 30.0
    # Measured on match_4/match_5: real passes implied at most 27 m/s from one
    # holder's feet to the next over the time between the two possessions;
    # the track jumping onto a look-alike implied 35-48 m/s.
    MAX_PASS_SPEED_MS = 30.0
    JUMP_SLACK_M = 1.0
    SETTLE_S = 2.0
    BRIEF_S = 0.3  # an opponent's touch this short does not win the ball...
    RESUMED_S = 1.5  # ...if a teammate of the passer has it within this
    BRIEF_MAX_SPEED_MS = 20.0

    def __init__(self):
        self.events: list[PassEvent] = []  # every pass, team not yet final
        self._counted: list[tuple[PassEvent, int | None, bool]] = []
        self._pending: list[PassEvent] = []
        self._last_ended: Possession | None = None
        self._holding: Possession | None = None  # current possession, if part of the chain
        self._holding_from: Possession | None = None  # ...and the one it was chained from
        self._holding_event: PassEvent | None = None  # ...and the pass it received
        self._last_from: Possession | None = None  # likewise for _last_ended
        self._last_event: PassEvent | None = None

    def on_possession_ended(self, ended: Possession) -> None:
        if ended is not self._holding:
            return
        self._holding = None
        event = self._holding_event
        if (
            event in self._pending
            and ended.end_t - ended.start_t < self.BRIEF_S
            and event.distance_m > self.BRIEF_MAX_SPEED_MS * max(event.t - event.from_t, 1e-3)
        ):
            self._pending.remove(event)
            self.events.remove(event)
            self._last_ended, self._last_from, self._last_event = self._holding_from, None, None
            return
        self._last_ended = ended
        self._last_from, self._last_event = self._holding_from, self._holding_event

    def on_possession_started(
        self, started: Possession, kicked_from: Possession | None = None
    ) -> None:
        """kicked_from: who was by the ball when its track began, if he could
        be the passer -- used only when there is no recent possession to
        chain from (a goal kick: the keeper steps back for his run-up and
        never holds the ball)."""
        if started.team == OTHER_TEAM:
            return
        prev = self._last_ended
        if (
            kicked_from is not None
            and kicked_from.team != OTHER_TEAM
            and (prev is None or started.start_t - prev.end_t > self.MAX_GAP_S)
        ):
            prev = kicked_from
        if self._only_touched(prev, started):
            self._pending.remove(self._last_event)
            self.events.remove(self._last_event)
            prev = self._last_from
        self._holding_from, self._holding_event = prev, None
        if prev is None or prev.player_id == started.player_id:
            self._holding = started
            return
        gap = started.start_t - prev.end_t
        distance = float(
            np.hypot(started.start_xy[0] - prev.end_xy[0], started.start_xy[1] - prev.end_xy[1])
        )
        if gap <= self.MAX_GAP_S and distance > self.JUMP_SLACK_M + self.MAX_PASS_SPEED_MS * gap:
            return  # faster than any kick: a jump, not a pass
        self._holding = started
        if gap > self.MAX_GAP_S:
            return
        completed = prev.team is not None and prev.team == started.team
        event = PassEvent(
            started.start_t,
            prev.player_id,
            started.player_id,
            prev.team,
            completed,
            distance,
            to_team=started.team,
            from_t=prev.end_t,
        )
        self.events.append(event)
        self._pending.append(event)
        self._holding_event = event
        self._last_ended = None

    def _only_touched(self, prev: Possession | None, started: Possession) -> bool:
        passer = self._last_from
        return (
            prev is not None
            and prev is self._last_ended
            and passer is not None
            and self._last_event in self._pending
            and prev.end_t - prev.start_t < self.BRIEF_S
            and started.start_t - prev.end_t <= self.RESUMED_S
            and passer.team is not None
            and passer.team == started.team != prev.team
        )

    def settle(self, t: float | None, team_at) -> None:
        """Counts every pending pass at least SETTLE_S old (all of them when t
        is None, at the end of the video). team_at(player_id, t) -> the team
        (0, 1, OTHER_TEAM or None) he played for around time t."""
        still_pending = []
        for e in self._pending:
            if t is not None and t - e.t < self.SETTLE_S:
                still_pending.append(e)
                continue
            team = self._team(team_at(e.from_player, e.from_t), e.team)
            to_team = self._team(team_at(e.to_player, e.t), e.to_team)
            if team is None and t is not None:
                still_pending.append(e)  # no teams yet
                continue
            self._counted.append((e, team, team is not None and team == to_team))
        self._pending = still_pending

    @staticmethod
    def _team(read, at_event):
        """The kit read around the pass, where it names a team; otherwise the
        side the player was put on at the time (a keeper, by position)."""
        return read if read in (0, 1) else at_event

    def counted(self) -> list[tuple[PassEvent, int | None, bool]]:
        """(event, passing team, completed) for every settled pass."""
        return self._counted

    def team_totals(self) -> dict:
        totals = {}
        for e, team, completed in self._counted:
            row = totals.setdefault(team, {"completed": 0, "lost": 0, "short": 0, "long": 0})
            row["completed" if completed else "lost"] += 1
            if completed:
                row["long" if e.distance_m >= self.LONG_PASS_M else "short"] += 1
        return totals


class TeamHistory:
    """Which team each player played for around a given moment: the majority
    of his team reads from WINDOW_BEFORE_S before it to PassCounter.SETTLE_S
    after it, within the same stint on screen.

    Neither the team at the moment of a pass nor the latest one is safe. A
    read goes wrong for seconds while a player stands against someone else
    (match_5's 6 read sky blue for 3 s beside a dark-coated steward on the
    touchline, right as he received a pass), and an id that swaps bodies in a
    tackle reads as the other kit from then on (match_4's 1, white until 49 s,
    yellow after) -- hence a window reaching well back. But an id that comes
    back after an absence over REAPPEAR_GAP_S may be on a new body (match_4's
    65: yellow until 93.8 s, a white player from 98.3 s; match_5's 6 in one
    run: City at 3 s, Tottenham from 25.8 s), so only reads since he came
    back count, and not those of his first REAPPEAR_GRACE_S while the colour
    average still carries the old kit, unless there are no others."""

    WINDOW_BEFORE_S = 10.0
    REAPPEAR_GAP_S = 1.0
    REAPPEAR_GRACE_S = 2.5
    KEEP_S = 15.0

    def __init__(self):
        self._reads: dict[int, list] = {}  # player_id -> [(t, team, stint start)], recent
        self._stint: dict[int, tuple[float, float]] = {}  # player_id -> (back since, last seen)

    def record(self, t: float, positions: dict, team_of) -> None:
        for pid in positions:
            since, last = self._stint.get(pid, (t, t))
            if t - last > self.REAPPEAR_GAP_S:
                since = t
            self._stint[pid] = (since, t)
            team = team_of(pid)
            if team is None:
                continue
            reads = self._reads.setdefault(pid, [])
            reads.append((t, team, since))
            if reads[0][0] < t - self.KEEP_S:
                self._reads[pid] = [r for r in reads if r[0] >= t - self.KEEP_S]

    def team_at(self, player_id: int, t: float):
        reads = self._reads.get(player_id, [])
        stints = [since for _, _, since in reads if since <= t]
        if not stints:
            return None
        stint = max(stints)
        near = [
            (rt, team)
            for rt, team, since in reads
            if since == stint and t - self.WINDOW_BEFORE_S <= rt <= t + PassCounter.SETTLE_S
        ]
        settled = [team for rt, team in near if rt - stint >= self.REAPPEAR_GRACE_S]
        votes = {}
        for team in settled or [team for _, team in near]:
            votes[team] = votes.get(team, 0) + 1
        return max(votes, key=votes.get) if votes else None


class BallAnalytics:
    """Per-result driver: candidates -> ball track -> possession -> passes.

    Keepers and officials share the classifier's OTHER_TEAM cluster (their
    kits match neither team), so a back pass to the keeper would read as
    "lost". A keeper is put on a side by where he stands: within KEEPER_ZONE_M
    of a goal line, on the team whose outfield players are on average nearer
    that goal (defenders are goal-side). Measured on match_5: at the back pass
    both teams were packed into one half, 20 and 25 m from the keeper -- too
    close for a nearest-team test -- but the defending team stood 5 m deeper.
    Anyone else in OTHER_TEAM (the referee in midfield) stays out of the pass
    chain.
    """

    KEEPER_ZONE_M = 25.0
    SIDE_MARGIN_M = 2.0
    MIN_SIDE_PLAYERS = 3
    KICKER_M = 4.0  # a player this close to where a ball track begins kicked it
    MIN_KICK_M = 5.0  # ...if the ball then travelled at least this far
    REACQUIRE_S = 0.3  # a sighting after coasting this long is a fresh start too

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
        self._kick: Possession | None = None  # who was by the ball when its track began
        self.teams = TeamHistory()

    def record(self, frame_id: int, candidates_pitch: list, positions: dict, team_of) -> None:
        if frame_id is None or frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        t = frame_id / self.fps
        self.results += 1
        coasted_s = None if self.ball is None else self.ball.coasting_s
        self.ball = self.tracker.update(t, candidates_pitch, positions)
        self.results_with_ball += self.ball is not None
        side_of = lambda pid: self._side(pid, positions, team_of)  # noqa: E731
        if self.ball is None:
            self._kick = None
        elif coasted_s is None:
            self._kick = self._kicker(t, positions, side_of)
        elif coasted_s >= self.REACQUIRE_S and self.ball.coasting_s == 0:
            # A goal kick's ball can be picked up by a track still coasting
            # on a look-alike, so a sighting after a gap is a start as well.
            self._kick = self._kicker(t, positions, side_of) or self._kick
        ended = self.possession.update(t, self.ball, positions, side_of)
        if ended is not None:
            self.passes.on_possession_ended(ended)
        current = self.possession.current
        holder = None if current is None else current.player_id
        if holder is not None and holder != self._last_holder:
            kick, self._kick = self._kick, None
            if kick is not None and (
                kick.player_id == holder
                or np.hypot(
                    kick.end_xy[0] - current.start_xy[0], kick.end_xy[1] - current.start_xy[1]
                )
                < self.MIN_KICK_M
            ):
                kick = None
            self.passes.on_possession_started(current, kicked_from=kick)
        self._last_holder = holder
        self.teams.record(t, positions, team_of)
        self.passes.settle(t, self.teams.team_at)

    def finish(self) -> None:
        """Counts the passes still settling when the video ends."""
        self.passes.settle(None, self.teams.team_at)

    def _kicker(self, t: float, positions: dict, side_of) -> Possession | None:
        nearest, dist = None, self.KICKER_M
        for pid, (x, y) in positions.items():
            d = float(np.hypot(x - self.ball.x, y - self.ball.y))
            if d <= dist:
                nearest, dist = pid, d
        if nearest is None:
            return None
        xy = positions[nearest]
        return Possession(nearest, side_of(nearest), t, t, xy, xy)

    def _side(self, player_id: int, positions: dict, team_of):
        team = team_of(player_id)
        if team != OTHER_TEAM or player_id not in positions:
            return team
        x = positions[player_id][0]
        if abs(x) < PITCH_LENGTH_M / 2 - self.KEEPER_ZONE_M:
            return OTHER_TEAM
        goal = np.sign(x)
        depth = []
        for side in (0, 1):
            xs = [p[0] for pid, p in positions.items() if team_of(pid) == side]
            if len(xs) < self.MIN_SIDE_PLAYERS:
                return OTHER_TEAM
            depth.append(goal * float(np.mean(xs)))
        if abs(depth[0] - depth[1]) < self.SIDE_MARGIN_M:
            return OTHER_TEAM
        return int(np.argmax(depth))

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
                    "team": team_name(team),
                    "completed": completed,
                    "distance_m": round(e.distance_m, 1),
                }
                for e, team, completed in self.passes.counted()
            ],
        }
