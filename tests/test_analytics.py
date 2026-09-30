"""Synthetic tests for scripts/analytics.py (Plan 3): projection, distance
that ignores jitter and out-of-frame gaps, sample gating, heatmap mass, and
team shape (what the camera can measure, officials, direction, possession)."""

import itertools
import json

import numpy as np
import pytest

from scripts.analytics import (
    FormationLines,
    MatchAnalytics,
    PitchProjector,
    PlayerTrace,
    TeamShapeAnalytics,
)

# pitch -> image: 10px per metre, origin at image (1000, 500); a pure affine
# homography so every expected value can be computed by hand.
H = np.array([[10.0, 0.0, 1000.0], [0.0, 10.0, 500.0], [0.0, 0.0, 1.0]])
FPS = 50.0


def _box_at(x_m, y_m, player_id, height_px=40):
    """A box whose bottom-centre is the given pitch position."""
    px, py = 1000 + 10 * x_m, 500 + 10 * y_m
    return (int(px - 8), int(py - height_px), int(px + 8), int(py), player_id)


def test_projector_maps_feet_to_pitch_metres():
    projector = PitchProjector(H)
    assert projector.to_pitch(1000, 500) == (0.0, 0.0)
    x, y = projector.to_pitch(1100, 450)
    assert (round(x, 6), round(y, 6)) == (10.0, -5.0)


def test_projector_rejects_off_pitch_and_missing_calibration():
    assert PitchProjector(None).to_pitch(1000, 500) is None
    assert not PitchProjector(None).available
    assert PitchProjector(H).to_pitch(1000 + 10 * 70, 500) is None  # 70m from centre: off the pitch


def test_straight_run_distance_and_speed():
    """5 m/s for 4 s along x -> ~20 m (the first half-second bucket has no
    predecessor, so a little under), top speed ~5 m/s."""
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 201):
        t = frame / FPS
        analytics.record(frame, [_box_at(5 * t, 0, 1)], set(), H, lambda pid: 0)
    analytics.finish()
    trace = analytics.players[1]
    assert 17.0 <= trace.distance_m <= 20.5
    assert 4.5 <= trace.top_speed_ms <= 5.5
    assert trace.team == 0
    assert abs(trace.tracked_s - 4.0) < 0.05


def test_pixel_jitter_on_a_standing_player_is_not_distance():
    """A stationary box whose bottom edge jitters by +-2px (a few cm) every
    frame: summing raw steps would give metres per second of fake motion."""
    analytics = MatchAnalytics(FPS)
    rng = np.random.default_rng(1)
    for frame in range(1, 251):
        jx, jy = rng.integers(-2, 3), rng.integers(-2, 3)
        box = (992 + jx, 460 + jy, 1008 + jx, 500 + jy, 7)
        analytics.record(frame, [box], set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.players[7].distance_m < 1.0
    assert analytics.players[7].top_speed_ms < 1.0


def test_time_out_of_frame_adds_no_distance():
    """Seen at x=0, gone for 3 s (camera panned away), seen again at x=30:
    the 30 m in between were not observed and must not be counted."""
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 26):
        analytics.record(frame, [_box_at(0, 0, 1)], set(), H, lambda pid: None)
    for frame in range(176, 201):
        analytics.record(frame, [_box_at(30, 0, 1)], set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.players[1].distance_m < 1.0
    assert abs(analytics.players[1].tracked_s - 1.0) < 0.1


def test_impossible_speed_is_dropped():
    """A lone player 40 m away half a second later is an identity error, not a sprint."""
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 26):
        analytics.record(frame, [_box_at(0, 0, 1)], set(), H, lambda pid: None)
    for frame in range(26, 51):
        analytics.record(frame, [_box_at(40, 0, 1)], set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.players[1].distance_m < 1.0


def test_gated_samples_are_dropped_and_counted():
    analytics = MatchAnalytics(FPS)
    analytics.record(1, [_box_at(0, 0, 1)], set(), None, lambda pid: None)  # no calibration yet
    analytics.record(2, [_box_at(0, 0, 1)], {1}, H, lambda pid: None)  # contaminated box
    analytics.record(3, [_box_at(80, 0, 1)], set(), H, lambda pid: None)  # off the pitch
    analytics.record(4, [_box_at(0, 0, 1)], set(), H, lambda pid: None)
    assert analytics.samples_dropped == {"no_homography": 1, "occluded": 1, "off_pitch": 1}
    assert analytics.players[1].samples == 1


def test_everyone_shifting_at_once_is_a_calibration_jump_not_motion():
    """Five players stand still, then all of them appear 2 m to the right in the
    next result: the pitch mapping moved, the players did not."""
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 51):
        boxes = [_box_at(5 * i, 0, i) for i in range(1, 6)]
        analytics.record(frame, boxes, set(), H, lambda pid: None)
    for frame in range(51, 101):
        boxes = [_box_at(5 * i + 2, 0, i) for i in range(1, 6)]
        analytics.record(frame, boxes, set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.calibration_shifts == 1
    assert all(trace.distance_m < 0.5 for trace in analytics.players.values())
    assert all(abs(trace.tracked_s - 2.0) < 0.1 for trace in analytics.players.values())


def test_one_player_moving_is_not_a_calibration_jump():
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 101):
        t = frame / FPS
        boxes = [_box_at(5 * i, 0, i) for i in range(2, 6)] + [_box_at(4 * t, 10, 1)]
        analytics.record(frame, boxes, set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.calibration_shifts == 0
    assert analytics.players[1].distance_m > 5.0


def test_motion_across_a_calibration_shift_is_kept_relative_to_the_pitch():
    """Four players stand still and one runs at 5 m/s; halfway, the mapping
    jumps 2 m for everyone. The runner's distance must be his run, not his run
    plus the jump, and the standers must still stand."""
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 201):
        t = frame / FPS
        offset = 2.0 if frame > 100 else 0.0
        boxes = [_box_at(5 * i + offset, 5, i) for i in range(2, 6)]
        boxes.append(_box_at(5 * t + offset, -5, 1))
        analytics.record(frame, boxes, set(), H, lambda pid: None)
    analytics.finish()
    assert analytics.calibration_shifts == 1
    assert 17.0 <= analytics.players[1].distance_m <= 21.0
    assert all(analytics.players[i].distance_m < 0.5 for i in range(2, 6))


def test_same_result_is_not_sampled_twice():
    """The display loop sees each worker result on several displayed frames."""
    analytics = MatchAnalytics(FPS)
    for _ in range(5):
        analytics.record(1, [_box_at(0, 0, 1)], set(), H, lambda pid: None)
    assert analytics.players[1].samples == 1


def test_heatmap_mass_sits_where_the_player_stood():
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 51):
        analytics.record(frame, [_box_at(20, -10, 3)], set(), H, lambda pid: 1)
    heat = analytics.players[3].heat
    row, col = np.unravel_index(heat.argmax(), heat.shape)
    assert (col, row) == (int(20 + 52.5), int(-10 + 34))
    assert abs(heat.sum() - 1.0) < 0.05  # 50 frames at 50fps = 1 s of presence
    assert analytics.team_heat(1).sum() == heat.sum()


def test_write_produces_stats_and_heatmaps(tmp_path):
    analytics = MatchAnalytics(FPS)
    for frame in range(1, 151):
        t = frame / FPS
        analytics.record(
            frame, [_box_at(3 * t, 0, 1), _box_at(-10, 5, 2)], set(), H, lambda pid: pid - 1
        )
    analytics.write(tmp_path, team_color=lambda team: (255, 0, 0))
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert [p["player_id"] for p in stats["players"]] == [1, 2]  # sorted by distance
    assert (tmp_path / "heatmap_player_1.png").exists()
    assert (tmp_path / "heatmap_team_0.png").exists()
    assert (tmp_path / "heatmap_team_1.png").exists()


def test_trace_window_constants_are_sane():
    assert PlayerTrace.WINDOW_S < PlayerTrace.MAX_GAP_S


# --- TeamShapeAnalytics -----------------------------------------------------

WHOLE_PITCH = (2000, 1000)  # H shows x in [-100, 100), y in [-50, 50): all of it
LEFT_HALF = (1000, 1000)  # x < 0 only: the camera ends at the halfway line


def _shape_record(shape, frame_id, teams, frame_size=WHOLE_PITCH, possession=None):
    """teams: team -> list of (x, y); records them as one result."""
    positions, team_of = {}, {}
    for team, points in teams.items():
        for i, xy in enumerate(points):
            pid = 100 * (team + 1) + i
            positions[pid] = xy
            team_of[pid] = team
    shape.record(frame_id, positions, team_of.get, H, frame_size, possession)


def _block(x0, y0, rows=2, cols=4, dx=10.0, dy=12.0):
    return [(x0 + r * dx, y0 + c * dy) for r in range(rows) for c in range(cols)]


def test_shape_measures_width_depth_area_of_a_visible_team():
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-30, -18), 1: _block(10, -18)})
    s = shape.latest[0]
    assert s.players == 8
    assert s.width_m == pytest.approx(36.0) and s.depth_m == pytest.approx(10.0)
    assert s.area_m2 == pytest.approx(360.0)


def test_shape_edge_cut_by_the_camera_is_not_measured():
    """The team reaches the halfway line and the picture ends there: its depth
    is unknown (players may stand beyond), its width is still seen."""
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-10, -18, rows=2, dx=9.0)}, frame_size=LEFT_HALF)
    s = shape.latest[0]
    assert s.depth_m is None and s.area_m2 is None
    assert s.width_m == pytest.approx(36.0)


def test_shape_edge_on_a_pitch_line_only_needs_the_line_in_view():
    """Nobody stands past the goal line, so a back line on it is closed as
    long as the line itself is in the picture."""
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-52.0, -18, dx=12.0)}, frame_size=LEFT_HALF)
    assert shape.latest[0].depth_m == pytest.approx(12.0)


def test_shape_too_few_players_is_no_shape():
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-30, -18, rows=1, cols=5)})
    assert 0 not in shape.latest


def test_shape_drops_an_official_on_the_touchline_but_keeps_a_wide_player():
    """A linesman read as this team, on the touchline 40 m from everyone, is
    dropped; a winger on the same line near his full-back is kept."""
    shape = TeamShapeAnalytics(FPS)
    team = _block(0, -18) + [(-40.0, -34.3)]  # linesman
    _shape_record(shape, 1, {0: team})
    assert shape.latest[0].players == 8 and shape.latest[0].width_m == pytest.approx(36.0)
    shape = TeamShapeAnalytics(FPS)
    team = _block(0, -18) + [(5.0, 33.8)]  # winger, 16 m from the nearest teammate
    _shape_record(shape, 1, {0: team})
    assert shape.latest[0].players == 9 and shape.latest[0].width_m == pytest.approx(51.8)


def test_shape_keeps_at_most_ten_outfield_players():
    shape = TeamShapeAnalytics(FPS)
    team = _block(0, -18, rows=3, cols=4)[:10] + [(-35.0, 0.0)]  # 11: the stray goes
    _shape_record(shape, 1, {0: team})
    assert shape.latest[0].players == 10
    assert shape.latest[0].depth_m == pytest.approx(20.0)


def test_shape_learns_which_goal_each_team_defends_and_its_line():
    """Team 1 stands goal-side (further +x) of team 0 throughout, so it
    defends +x; its line is where its last outfield man stands, from that goal."""
    shape = TeamShapeAnalytics(FPS)
    attackers = _block(0, -18)
    defenders = [(30.0, -18.0), (30.0, -6.0), (32.0, 6.0), (32.0, 18.0)] + _block(10, -18, rows=1)
    for frame in range(1, int(3 * FPS)):
        _shape_record(shape, frame, {0: attackers, 1: defenders})
    assert shape.own_goal == {1: 1, 0: -1}
    assert shape.latest[1].line_m == pytest.approx(52.5 - 32.0)


def test_shape_line_waits_for_the_direction():
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-30, -18), 1: _block(10, -18)})
    assert shape.own_goal is None and shape.latest[1].line_m is None


def test_shape_stats_split_by_possession():
    shape = TeamShapeAnalytics(FPS)
    frame = 0
    for team_on_ball, width in ((0, 12.0), (1, 8.0)):  # wide with the ball, narrow without
        for _ in range(30):
            frame += int(FPS * TeamShapeAnalytics.SAMPLE_S)
            _shape_record(
                shape,
                frame,
                {0: _block(-30, -18, dy=width), 1: _block(10, -18)},
                possession=team_on_ball,
            )
    stats = shape.team_stats(0)["width_m"]
    assert stats["in_possession"] == pytest.approx(36.0)
    assert stats["out_of_possession"] == pytest.approx(24.0)
    assert stats["all_results"] == 60


def test_shape_display_value_holds_briefly_then_clears():
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-30, -18)})
    assert shape.display_value(0, "width_m") == pytest.approx(36.0)
    later = 1 / FPS + TeamShapeAnalytics.DISPLAY_HOLD_S + 0.1
    assert shape.display_value(0, "width_m", t=later) is None


def test_shape_write(tmp_path):
    shape = TeamShapeAnalytics(FPS)
    for frame in range(1, 30):
        _shape_record(shape, frame, {0: _block(-30, -18), 1: _block(10, -18)}, possession=0)
    shape.write(tmp_path, team_name=lambda team: f"team {team}")
    data = json.loads((tmp_path / "shape.json").read_text())
    assert data["teams"]["team 0"]["width_m"]["all"] == 36.0
    assert data["series"][0]["0"]["phase"] == "in"


# --- FormationLines ----------------------------------------------------------

FOUR_FOUR_TWO = {
    1: 0.0,
    2: 0.5,
    3: 1.0,
    4: 0.2,
    5: 12.0,
    6: 12.5,
    7: 13.0,
    8: 11.5,
    9: 24.0,
    10: 25.0,
}


def _lines_over(lines, depths_at, seconds, team=0, t0=0.0):
    """Feeds depths_at(t) every 0.1 s for `seconds`; returns the last lines."""
    out = None
    for i in range(int(seconds * 10)):
        t = t0 + i / 10
        out = lines.update(t, team, depths_at(t))
    return out


def test_formation_lines_cut_where_the_gaps_are():
    lines = FormationLines()
    defence, midfield, attack = _lines_over(lines, lambda t: FOUR_FOUR_TWO, 1.0)
    assert set(defence) == {1, 2, 3, 4} and set(midfield) == {5, 6, 7, 8}
    assert set(attack) == {9, 10}


def test_formation_lines_do_not_flip_for_a_player_between_two_lines():
    """A full back hovering across the point where the best cut changes
    (6.45 m here), a metre either side of it every half second, stays in one
    line instead of switching back and forth (5 switches without hysteresis)."""
    lines = FormationLines()
    _lines_over(lines, lambda t: FOUR_FOUR_TWO, 2.0)
    seen_in = []
    for i in range(60):
        t = 2.0 + i / 10
        hover = {**FOUR_FOUR_TWO, 4: 7.5 if (i // 5) % 2 else 5.5}
        defence, _, _ = lines.update(t, 0, hover)
        seen_in.append(4 in defence)
    switches = sum(a != b for a, b in itertools.pairwise(seen_in))
    assert switches <= 1


def test_formation_lines_follow_a_player_who_really_moves_up():
    lines = FormationLines()
    _lines_over(lines, lambda t: FOUR_FOUR_TWO, 2.0)
    pushed = {**FOUR_FOUR_TWO, 4: 12.3}  # the full back stays up: 3-5-2
    defence, midfield, _ = _lines_over(lines, lambda t: pushed, 4.0, t0=2.0)
    assert 4 in midfield and len(defence) == 3


def test_formation_lines_need_enough_players_and_forget_the_gone():
    lines = FormationLines()
    few = {pid: d for pid, d in FOUR_FOUR_TWO.items() if pid <= 5}
    assert _lines_over(lines, lambda t: few, 1.0) is None
    _lines_over(lines, lambda t: FOUR_FOUR_TWO, 1.0, t0=1.0)
    without_ten = {pid: d for pid, d in FOUR_FOUR_TWO.items() if pid != 10}
    defence, midfield, attack = _lines_over(lines, lambda t: without_ten, 3.0, t0=2.0)
    assert 10 not in defence + midfield + attack


def test_shape_formation_label_and_lines_of_a_whole_team():
    shape = TeamShapeAnalytics(FPS)
    back, mid, front = -40.0, -28.0, -16.0  # team 0 defends -x: a 4-4-2
    team0 = [(back, y) for y in (-20, -7, 7, 20)] + [(mid, y) for y in (-20, -7, 7, 20)]
    team0 += [(front, -5.0), (front, 5.0)]
    team1 = [(x + 30.0, y) for x, y in team0]
    for frame in range(1, int(4 * FPS), 5):
        _shape_record(shape, frame, {0: team0, 1: team1})
    assert shape.latest[0].formation == "4-4-2"


def test_shape_possession_is_held_while_the_ball_travels():
    shape = TeamShapeAnalytics(FPS)
    _shape_record(shape, 1, {0: _block(-30, -18)}, possession=0)
    _shape_record(shape, 100, {0: _block(-30, -18)})  # 2 s later, ball in flight
    assert shape.in_possession == 0 and shape.defending_team == 1
    for frame in (150, 155, 160):  # a touch read on the other side for 0.2 s
        _shape_record(shape, frame, {0: _block(-30, -18)}, possession=1)
    assert shape.defending_team == 1  # is not a change of possession yet
    for frame in range(165, 215, 5):
        _shape_record(shape, frame, {0: _block(-30, -18)}, possession=1)
    assert shape.defending_team == 0  # it lasted POSSESSION_SWITCH_S
    late = frame + int((TeamShapeAnalytics.POSSESSION_HOLD_S + 1) * FPS)
    _shape_record(shape, late, {0: _block(-30, -18)})
    assert shape.defending_team is None
