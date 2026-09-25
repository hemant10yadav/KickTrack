"""Synthetic tests for scripts/ball.py (Plan 3.2): ball tracking through gaps
and look-alikes, possession, and pass counting."""

from scripts.ball import BallAnalytics, BallTracker, PassCounter, PossessionTracker

FPS = 50.0


def _feed(tracker, path, fps=FPS, extra=None):
    """path: frame -> [(x, y, conf), ...] or None (no candidates)."""
    states = {}
    for frame in sorted(path):
        cands = list(path[frame] or [])
        if extra is not None:
            cands += extra(frame)
        states[frame] = tracker.update(frame / fps, cands)
    return states


def test_ball_needs_two_consistent_sightings_to_start():
    tracker = BallTracker()
    assert tracker.update(1 / FPS, [(10.0, 0.0, 0.5)]) is None
    assert tracker.update(2 / FPS, [(10.2, 0.0, 0.5)]) is not None


def test_a_single_stray_blob_never_starts_a_track():
    tracker = BallTracker()
    tracker.update(1 / FPS, [(10.0, 0.0, 0.9)])
    assert tracker.update(2 / FPS, [(40.0, 20.0, 0.9)]) is None  # nowhere near the first


def test_rolling_ball_coasts_through_a_gap_and_reacquires_where_it_should_be():
    """10 m/s along x, unseen for 0.5 s, then seen again 5 m further on."""
    tracker = BallTracker()
    path = {f: [(10.0 * f / FPS, 0.0, 0.4)] for f in range(1, 26)}
    for f in range(26, 51):
        path[f] = None
    states = _feed(tracker, path)
    coasting = states[50]
    assert coasting is not None and coasting.coasting_s > 0.4
    assert abs(coasting.x - 10.0) < 1.5  # predicted forward, not frozen at 5 m

    state = tracker.update(51 / FPS, [(10.2, 0.0, 0.4)])
    assert state.coasting_s == 0.0 and abs(state.x - 10.2) < 0.7


def test_ball_is_lost_after_a_long_gap():
    tracker = BallTracker()
    path = {f: [(0.0, 0.0, 0.4)] for f in range(1, 11)}
    for f in range(11, 100):
        path[f] = None
    states = _feed(tracker, path)
    assert states[99] is None


def test_far_lookalike_is_rejected_while_the_ball_is_tracked():
    """A confident static blob 30 m away (a head, a sock) must not steal the
    track from a ball it could not have reached."""
    tracker = BallTracker()
    path = {f: [(2.0 * f / FPS, 0.0, 0.2)] for f in range(1, 51)}
    states = _feed(tracker, path, extra=lambda f: [(30.0, 20.0, 0.9)] if f > 2 else [])
    assert abs(states[50].x - 2.0) < 1.0 and abs(states[50].y) < 1.0
    assert tracker.rejected + tracker.static_rejected >= 46


def test_static_blob_with_nobody_near_is_a_fixed_feature_not_the_ball():
    """A penalty spot or a touchline object shows up in the same place every
    frame; the ball in play never does without a player at it."""
    tracker = BallTracker()
    players = {1: (30.0, 0.0)}
    states = _feed(tracker, {f: [(0.0, 0.0, 0.6)] for f in range(1, 151)}, extra=lambda f: [])
    assert states[150] is None or tracker.static_rejected == 0  # first pass, no players given
    tracker = BallTracker()
    for f in range(1, 151):
        state = tracker.update(f / FPS, [(0.0, 0.0, 0.6)], players)
    assert state is None and tracker.static_rejected > 0


def test_static_ball_at_a_players_feet_is_still_the_ball():
    tracker = BallTracker()
    for f in range(1, 151):
        state = tracker.update(f / FPS, [(0.0, 0.0, 0.6)], {1: (0.8, 0.0)})
    assert state is not None and tracker.static_rejected == 0


def test_kicked_ball_is_followed():
    """Rolling at 2 m/s, then kicked to 20 m/s: the gate must admit the jump."""
    tracker = BallTracker()
    path = {f: [(2.0 * f / FPS, 0.0, 0.3)] for f in range(1, 26)}
    x25 = 2.0 * 25 / FPS
    for f in range(26, 51):
        path[f] = [(x25 + 20.0 * (f - 25) / FPS, 0.0, 0.3)]
    states = _feed(tracker, path)
    assert abs(states[50].x - (x25 + 10.0)) < 1.5
    assert states[50].vx > 12.0


def _run_possession(sequence, team_of=lambda pid: 0):
    """sequence: list of (ball_xy or None, {pid: (x, y)}) per result at 50 fps."""
    tracker = PossessionTracker()
    ended = []
    for i, (ball_xy, positions) in enumerate(sequence):
        t = (i + 1) / FPS
        ball = None
        if ball_xy is not None:
            from scripts.ball import BallState

            ball = BallState(t, ball_xy[0], ball_xy[1], 0.0, 0.0, 0.0, 0.5)
        e = tracker.update(t, ball, positions, team_of)
        if e is not None:
            ended.append(e)
    return tracker, ended


def test_possession_needs_the_ball_held_not_just_passing_by():
    players = {1: (0.0, 0.0)}
    tracker, _ = _run_possession([((0.5, 0.0), players)] * 2)
    assert tracker.current is None
    tracker, _ = _run_possession([((0.5, 0.0), players)] * 3)
    assert tracker.current is not None and tracker.current.player_id == 1


def test_possession_ends_when_the_ball_leaves_and_records_where():
    players = {1: (0.0, 0.0)}
    seq = [((0.5, 0.0), players)] * 5 + [((10.0, 0.0), players)] * 3
    tracker, ended = _run_possession(seq)
    assert tracker.current is None and len(ended) == 1
    assert ended[0].player_id == 1 and ended[0].end_xy == (0.0, 0.0)


def test_pass_between_teammates_is_completed_and_measured():
    passes = PassCounter()
    a = _run_possession([((0.5, 0.0), {1: (0.0, 0.0)})] * 5 + [((10.0, 0.0), {1: (0.0, 0.0)})] * 3)[
        1
    ][0]
    passes.on_possession_ended(a)
    from scripts.ball import Possession

    b = Possession(2, 0, a.end_t + 1.0, a.end_t + 1.0, (20.0, 0.0), (20.0, 0.0))
    passes.on_possession_started(b)
    assert len(passes.events) == 1
    e = passes.events[0]
    assert e.completed and e.from_player == 1 and e.to_player == 2
    assert abs(e.distance_m - 20.0) < 0.01
    assert passes.team_totals() == {0: {"completed": 1, "lost": 0, "short": 1, "long": 0}}


def test_pass_to_an_opponent_is_lost_and_a_long_one_is_long():
    from scripts.ball import Possession

    passes = PassCounter()
    passes.on_possession_ended(Possession(1, 0, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0)))
    passes.on_possession_started(Possession(2, 1, 2.0, 2.0, (35.0, 0.0), (35.0, 0.0)))
    passes.on_possession_ended(Possession(2, 1, 2.0, 3.0, (35.0, 0.0), (35.0, 0.0)))
    passes.on_possession_started(Possession(3, 1, 4.0, 4.0, (0.0, 0.0), (0.0, 0.0)))
    totals = passes.team_totals()
    assert totals[0] == {"completed": 0, "lost": 1, "short": 0, "long": 0}
    assert totals[1] == {"completed": 1, "lost": 0, "short": 0, "long": 1}


def test_a_possession_too_long_after_the_last_is_not_a_pass():
    from scripts.ball import Possession

    passes = PassCounter()
    passes.on_possession_ended(Possession(1, 0, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0)))
    passes.on_possession_started(Possession(2, 0, 9.0, 9.0, (5.0, 0.0), (5.0, 0.0)))
    assert passes.events == []


def test_end_to_end_pass_from_candidates():
    """Player 1 holds the ball, kicks it 20 m to teammate 2 who holds it."""
    analytics = BallAnalytics(FPS)
    positions = {1: (0.0, 0.0), 2: (20.0, 0.0)}
    for frame in range(1, 21):  # at player 1's feet
        analytics.record(frame, [(0.4, 0.0, 0.4)], positions, lambda pid: 0)
    for frame in range(21, 61):  # in flight, 25 m/s -> 20 m in 0.8 s
        x = min(20.0, 0.4 + 25.0 * (frame - 20) / FPS)
        analytics.record(frame, [(x, 0.0, 0.3)], positions, lambda pid: 0)
    for frame in range(61, 91):  # at player 2's feet
        analytics.record(frame, [(19.6, 0.0, 0.4)], positions, lambda pid: 0)
    assert analytics.holder == 2
    assert analytics.passes.team_totals() == {0: {"completed": 1, "lost": 0, "short": 1, "long": 0}}
    assert "1 passes (1 short, 0 long), 0 lost" in analytics.summary(team_name=lambda t: "white")
