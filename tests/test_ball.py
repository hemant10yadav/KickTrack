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
    passes.on_possession_started(a)
    passes.on_possession_ended(a)
    from scripts.ball import Possession

    b = Possession(2, 0, a.end_t + 1.0, a.end_t + 1.0, (20.0, 0.0), (20.0, 0.0))
    passes.on_possession_started(b)
    passes.settle(None, lambda pid, t: None)
    assert len(passes.events) == 1
    e = passes.events[0]
    assert e.completed and e.from_player == 1 and e.to_player == 2
    assert abs(e.distance_m - 20.0) < 0.01
    assert passes.team_totals() == {0: {"completed": 1, "lost": 0, "short": 1, "long": 0}}


def test_pass_to_an_opponent_is_lost_and_a_long_one_is_long():
    from scripts.ball import Possession

    passes = PassCounter()
    for p in (
        Possession(1, 0, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0)),
        Possession(2, 1, 2.5, 3.0, (35.0, 0.0), (35.0, 0.0)),  # 35 m in 1.5 s
        Possession(3, 1, 5.0, 5.0, (0.0, 0.0), (0.0, 0.0)),
    ):
        passes.on_possession_started(p)
        passes.on_possession_ended(p)
    passes.settle(None, lambda pid, t: None)
    totals = passes.team_totals()
    assert totals[0] == {"completed": 0, "lost": 1, "short": 0, "long": 0}
    assert totals[1] == {"completed": 1, "lost": 0, "short": 0, "long": 1}


def test_a_possession_too_long_after_the_last_is_not_a_pass():
    from scripts.ball import Possession

    passes = PassCounter()
    first = Possession(1, 0, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0))
    passes.on_possession_started(first)
    passes.on_possession_ended(first)
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
    assert analytics.passes.team_totals() == {}  # not settled yet
    analytics.finish()
    assert analytics.passes.team_totals() == {0: {"completed": 1, "lost": 0, "short": 1, "long": 0}}
    assert "1 passes (1 short, 0 long), 0 lost" in analytics.summary(team_name=lambda t: "white")


def _possession(pid, team, t0, t1, xy):
    from scripts.ball import Possession

    return Possession(pid, team, t0, t1, xy, xy)


def test_a_possession_the_ball_could_not_have_reached_is_a_jump_not_a_pass():
    """match_5 at 29.9 s: the track leapt 22 m in 0.46 s onto a look-alike by
    another player, and back, while one player dribbled alone."""
    passes = PassCounter()
    a = _possession(1, 0, 0.0, 1.0, (0.0, 0.0))
    passes.on_possession_started(a)
    passes.on_possession_ended(a)
    jump = _possession(2, 0, 1.4, 1.5, (22.0, 0.0))
    passes.on_possession_started(jump)
    passes.on_possession_ended(jump)
    passes.on_possession_started(_possession(1, 0, 3.0, 3.0, (3.0, 0.0)))
    assert passes.events == []


def test_the_referee_is_not_part_of_a_pass():
    from scripts.player import OTHER_TEAM

    passes = PassCounter()
    a = _possession(1, 0, 0.0, 1.0, (0.0, 0.0))
    passes.on_possession_started(a)
    passes.on_possession_ended(a)
    ref = _possession(9, OTHER_TEAM, 1.5, 1.8, (5.0, 0.0))
    passes.on_possession_started(ref)
    passes.on_possession_ended(ref)
    passes.on_possession_started(_possession(2, 0, 2.5, 2.5, (10.0, 0.0)))
    passes.settle(None, lambda pid, t: None)
    assert [(e.from_player, e.to_player, team, done) for e, team, done in passes.counted()] == [
        (1, 2, 0, True)
    ]


def test_a_pass_is_counted_only_once_settled_and_with_the_teams_around_it():
    """Player 2's id reads as the wrong kit at the pass and the right one
    after it; the count waits, then books it for the right team."""
    passes = PassCounter()
    a = _possession(1, 0, 0.0, 1.0, (0.0, 0.0))
    passes.on_possession_started(a)
    passes.on_possession_ended(a)
    passes.on_possession_started(_possession(2, 1, 1.5, 1.5, (10.0, 0.0)))
    team_at = lambda pid, t: 0  # noqa: E731 -- both white around the pass
    passes.settle(2.0, team_at)
    assert passes.counted() == []
    passes.settle(1.5 + PassCounter.SETTLE_S, team_at)
    assert [(team, done) for _, team, done in passes.counted()] == [(0, True)]


def test_team_history_ignores_an_id_just_back_and_a_later_body_swap():
    """match_5's 6 came back after 22 s on a new body and read as his old kit
    for ~2 s; match_4's 1 swapped onto an opponent in a tackle."""
    from scripts.ball import TeamHistory

    history = TeamHistory()

    def play(t0, t1, team):
        for step in range(round(t0 * 10), round(t1 * 10)):
            history.record(step / 10, {7: (0.0, 0.0)}, lambda pid: team)

    play(0.0, 3.0, 1)  # another body
    play(6.0, 7.5, 1)  # back at 6 s as a white, still read as the old kit...
    play(7.5, 12.0, 0)  # ...until the colour average catches up
    assert history.team_at(7, 7.0) == 0  # a pass at 7 s: the stale reads are ignored
    play(12.0, 20.0, 1)  # swapped onto an opponent at 12 s
    assert history.team_at(7, 10.0) == 0  # a pass before the swap keeps his kit then


def _run_analytics(frames, team_of, fps=FPS):
    analytics = BallAnalytics(fps)
    for frame, (cands, positions) in enumerate(frames, start=1):
        analytics.record(frame, cands, positions, team_of)
    analytics.finish()
    return analytics


def _defenders_and_attackers():
    """Team 1 defends the +x goal (its players deeper), team 0 attacks."""
    positions = {pid: (30.0 + pid, -10.0 + 4 * pid) for pid in range(1, 5)}  # team 1
    positions.update({pid: (10.0 + pid, -10.0 + 4 * (pid - 10)) for pid in range(11, 15)})
    return positions


def test_a_back_pass_to_the_keeper_is_completed_for_the_defending_team():
    from scripts.player import OTHER_TEAM

    positions = _defenders_and_attackers()
    positions[99] = (48.0, 0.0)  # keeper, in OTHER_TEAM's colour
    team_of = lambda pid: OTHER_TEAM if pid == 99 else (1 if pid < 10 else 0)  # noqa: E731
    at = lambda xy: [(xy[0] + 0.4, xy[1], 0.4)]  # noqa: E731
    frames = [(at(positions[1]), positions)] * 20
    for i in range(1, 40):  # 1 -> keeper, 17 m at ~22 m/s
        x = positions[1][0] + (48.0 - positions[1][0]) * i / 40
        y = positions[1][1] * (1 - i / 40)
        frames.append(([(x, y, 0.3)], positions))
    frames += [(at(positions[99]), positions)] * 20
    analytics = _run_analytics(frames, team_of)
    assert [
        (e.from_player, e.to_player, team, done) for e, team, done in analytics.passes.counted()
    ] == [(1, 99, 1, True)]


def test_a_goal_kick_is_a_pass_from_the_keeper_who_never_held_the_ball():
    """The keeper steps back for his run-up: the ball's track only begins as
    it is struck, with him 3 m away."""
    from scripts.player import OTHER_TEAM

    positions = _defenders_and_attackers()
    positions[99] = (47.0, 0.0)
    team_of = lambda pid: OTHER_TEAM if pid == 99 else (1 if pid < 10 else 0)  # noqa: E731
    frames = [([], positions)] * 10
    target = positions[2]
    for i in range(0, 30):  # struck from (44, 0) towards player 2, ~15 m/s
        x = 44.0 + (target[0] - 44.0) * i / 30
        y = target[1] * i / 30
        frames.append(([(x, y, 0.3)], positions))
    frames += [([(target[0] + 0.4, target[1], 0.4)], positions)] * 20
    analytics = _run_analytics(frames, team_of)
    assert [
        (e.from_player, e.to_player, team, done) for e, team, done in analytics.passes.counted()
    ] == [(99, 2, 1, True)]
