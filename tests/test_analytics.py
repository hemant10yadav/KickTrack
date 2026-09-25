"""Synthetic tests for scripts/analytics.py (Plan 3): projection, distance
that ignores jitter and out-of-frame gaps, sample gating, and heatmap mass."""

import json

import numpy as np

from scripts.analytics import MatchAnalytics, PitchProjector, PlayerTrace

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
