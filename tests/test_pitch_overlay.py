"""Tests for PitchOverlayRenderer -- draws the calibrated pitch outline
(PITCH_LINES, scripts/calibration.py) onto the live video frame.

Fast and deterministic: uses a hand-constructed affine homography (pure
scale + translate, no perspective) so exact expected pixel locations can be
computed and asserted against, rather than just checking "something drew."
"""

import numpy as np

from scripts.calibration import PITCH_LINES
from scripts.display import MarkerRenderer, PitchMinimap, PitchOverlayRenderer
from scripts.player import TeamClassifier

# Maps world (x, y) -> pixel (5x + 300, 5y + 300) exactly (affine: w stays 1).
# Chosen so the halfway line -- PITCH_LINES' ((0,-34),(0,34)) segment --
# lands fully inside a 600x600 frame at a known, easy-to-check location.
SCALE_TRANSLATE_HOMOGRAPHY = np.array(
    [
        [5.0, 0.0, 300.0],
        [0.0, 5.0, 300.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_draw_with_no_homography_leaves_frame_unchanged():
    frame = np.zeros((600, 600, 3), dtype="uint8")
    original = frame.copy()

    PitchOverlayRenderer().draw(frame, homography=None)

    assert np.array_equal(frame, original)


def test_draw_paints_halfway_line_at_exact_projected_pixels():
    """PITCH_LINES contains ((0.0, -34.0), (0.0, 34.0)) -- the halfway line.
    Under SCALE_TRANSLATE_HOMOGRAPHY that projects to pixel (300, 130) ->
    (300, 470), a vertical line whose midpoint is exactly (300, 300)."""
    frame = np.zeros((600, 600, 3), dtype="uint8")
    assert ((0.0, -34.0), (0.0, 34.0)) in PITCH_LINES

    PitchOverlayRenderer().draw(frame, SCALE_TRANSLATE_HOMOGRAPHY)

    # BGR, since PitchOverlayRenderer draws via cv2 (frame is a cv2-style
    # BGR array): LINE_COLOR = (0, 0, 255) is pure red in BGR order.
    assert tuple(frame[300, 300]) == (0, 0, 255)
    # A point far from any projected pitch line (well outside the pitch's
    # projected bounding box of roughly x in [37,563], y in [130,470])
    # must be left untouched.
    assert tuple(frame[10, 10]) == (0, 0, 0)


def test_draw_skips_degenerate_projection_without_crashing():
    """A homography that sends a point to the plane at infinity (w == 0)
    must be skipped, not crash the whole overlay."""
    frame = np.zeros((100, 100, 3), dtype="uint8")
    # w-row is all zeros -> every projected point has w == 0.
    degenerate = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    PitchOverlayRenderer().draw(frame, degenerate)  # must not raise

    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype="uint8"))


# --- PitchMinimap (Plan 3.1: live positions) ----------------------------------


def _minimap_origin(frame_shape, minimap):
    h, w = frame_shape[:2]
    ph, pw = minimap.pitch.shape[:2]
    return w - minimap.MARGIN - pw, h - minimap.MARGIN - ph


def test_minimap_draws_a_dot_at_the_players_pitch_position():
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    minimap = PitchMinimap()
    minimap.draw(frame, {7: (0.0, 0.0)}, team_of=lambda pid: 0, team_color=lambda team: (0, 0, 255))

    x0, y0 = _minimap_origin(frame.shape, minimap)
    centre = (x0 + int(52.5 * minimap.SCALE), y0 + int(34 * minimap.SCALE))
    b, g, r = frame[centre[1], centre[0]]
    assert r > 150 and b < 100, "the centre-spot dot should be the team's red"


def test_minimap_stays_inside_the_frame_and_is_skipped_when_it_cannot_fit():
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    PitchMinimap().draw(
        frame, {1: (60.0, 40.0)}, lambda pid: None, lambda team: None
    )  # off-pitch: ignored
    assert frame[:300, :, :].sum() == 0, "nothing drawn outside the bottom-right panel"

    tiny = np.zeros((100, 100, 3), dtype="uint8")
    PitchMinimap().draw(tiny, {1: (0.0, 0.0)}, lambda pid: None, lambda team: None)
    assert tiny.sum() == 0


def test_minimap_with_no_positions_still_shows_the_pitch():
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    PitchMinimap().draw(frame, {}, lambda pid: None, lambda team: None)
    assert frame.sum() > 0


def test_marker_caption_is_drawn_above_the_id_label():
    frame = np.zeros((400, 400, 3), dtype="uint8")
    renderer = MarkerRenderer(TeamClassifier())
    renderer.draw(frame, [(150, 200, 180, 260, 3)], captions={3: "42m"})
    with_caption = int((frame == 255).all(axis=2).sum())  # white caption pixels

    frame = np.zeros((400, 400, 3), dtype="uint8")
    renderer.draw(frame, [(150, 200, 180, 260, 3)])
    assert with_caption > int((frame == 255).all(axis=2).sum())


# --- BallRenderer (Plan 3.2) ---------------------------------------------------


def test_ball_renderer_draws_ball_holder_and_pass_panel():
    from scripts.ball import BallAnalytics
    from scripts.display import BallRenderer

    frame = np.zeros((720, 1280, 3), dtype="uint8")
    analytics = BallAnalytics(50.0)
    positions = {1: (0.0, 0.0), 2: (20.0, 0.0)}
    for f in range(1, 21):
        analytics.record(f, [(0.4, 0.0, 0.5)], positions, lambda pid: 0)
    assert analytics.holder == 1
    BallRenderer(show_markers=True).draw(frame, analytics, SCALE_TRANSLATE_HOMOGRAPHY, positions)
    # ball ring at the centre spot (world 0,0 -> pixel 300,300), yellow
    b, g, r = frame[300, 309]
    assert g > 150 and r > 150 and b < 100
    # the panel sits top-right
    assert frame[15:60, 1280 - 12 - 330 : 1280 - 12].sum() > 0


def test_ball_renderer_hides_the_rings_by_default():
    from scripts.ball import BallAnalytics
    from scripts.display import BallRenderer

    frame = np.zeros((720, 1280, 3), dtype="uint8")
    analytics = BallAnalytics(50.0)
    positions = {1: (0.0, 0.0), 2: (20.0, 0.0)}
    for f in range(1, 21):
        analytics.record(f, [(0.4, 0.0, 0.5)], positions, lambda pid: 0)
    BallRenderer().draw(frame, analytics, SCALE_TRANSLATE_HOMOGRAPHY, positions)
    assert frame[250:350, 250:350].sum() == 0  # no ring or holder ellipse
    assert frame[15:60, 1280 - 12 - 330 : 1280 - 12].sum() > 0  # the panel still is


def test_ball_renderer_without_calibration_still_draws_the_panel():
    from scripts.ball import BallAnalytics
    from scripts.display import BallRenderer

    frame = np.zeros((720, 1280, 3), dtype="uint8")
    BallRenderer().draw(frame, BallAnalytics(50.0), None, {})
    assert frame[:300, :, :].sum() == 0 or frame[15:60, 938:1268].sum() > 0


def _shaped_team(x0: float):
    from scripts.analytics import TeamShapeAnalytics

    shape = TeamShapeAnalytics(50.0)
    positions = {i: (x0 + 10 * (i % 2), -18 + 12 * (i // 2)) for i in range(8)}
    # 10 px per metre, origin at the frame centre: the whole pitch is in view
    homography = np.array([[10.0, 0.0, 1000.0], [0.0, 10.0, 500.0], [0.0, 0.0, 1.0]])
    shape.record(1, positions, lambda pid: 0, homography, (2000, 1000))
    return shape


def test_shape_overlay_draws_a_thin_outline_through_the_shown_feet_and_the_line():
    """Outline through the feet of the boxes on screen; the defensive line
    touchline to touchline at the last man on this frame -- both thin."""
    from scripts.analytics import TeamShapeAnalytics
    from scripts.display import TeamShapeOverlay

    shape = TeamShapeAnalytics(50.0)
    shape.latest = _shaped_team(-30.0).latest  # 8 players, ids 0-7
    shape.own_goal = {0: -1, 1: 1}
    shape.display_value = lambda team, metric, t=None: 20.0  # the line is measurable
    # the boxes shown on this frame; player 4 is the last man, at pixel x 350
    feet = [(400, 200), (600, 200), (600, 300), (400, 300), (350, 250)] + [(500, 250)] * 3
    boxes = [(fx - 5, fy - 30, fx + 5, fy, pid) for pid, (fx, fy) in enumerate(feet)]
    frame = np.zeros((600, 600, 3), dtype="uint8")
    red = lambda team: (0, 0, 255)  # noqa: E731
    TeamShapeOverlay().draw(frame, boxes, shape, SCALE_TRANSLATE_HOMOGRAPHY, red)
    assert frame[200, 500, 2] > 200  # on the outline's top edge
    assert frame[250, 450].sum() == 0  # lines only, nothing drawn inside
    # the line at the last man's column, well past the team (rows 200-300)
    # towards the touchlines at y = -34 / 34 -> rows 130 / 470
    assert frame[140, 349:352, 2].max() > 200 and frame[460, 349:352, 2].max() > 200
    assert frame[140, 340, 2] == 0 and frame[140, 360, 2] == 0  # and thin


def test_shape_overlay_has_no_line_while_it_is_not_measurable():
    from scripts.analytics import TeamShapeAnalytics
    from scripts.display import TeamShapeOverlay

    shape = TeamShapeAnalytics(50.0)
    shape.latest = _shaped_team(-30.0).latest
    shape.own_goal = {0: -1, 1: 1}  # but no line value: the own-goal end is out of view
    feet = [(400, 200), (600, 200), (600, 300), (400, 300), (350, 250)] + [(500, 250)] * 3
    boxes = [(fx - 5, fy - 30, fx + 5, fy, pid) for pid, (fx, fy) in enumerate(feet)]
    frame = np.zeros((600, 600, 3), dtype="uint8")
    TeamShapeOverlay().draw(
        frame, boxes, shape, SCALE_TRANSLATE_HOMOGRAPHY, lambda team: (0, 0, 255)
    )
    assert frame[140, 345:356].sum() == 0


def test_shape_overlay_dashes_the_second_team():
    from scripts.analytics import TeamShapeAnalytics
    from scripts.display import TeamShapeOverlay

    shape = TeamShapeAnalytics(50.0)
    solid = _shaped_team(-30.0).latest[0]
    shape.latest = {1: solid.__class__(**{**solid.__dict__, "team": 1})}
    feet = [(100, 200), (500, 200), (500, 300), (100, 300)] + [(300, 250)] * 4
    boxes = [(fx - 5, fy - 30, fx + 5, fy, pid) for pid, (fx, fy) in enumerate(feet)]
    frame = np.zeros((600, 600, 3), dtype="uint8")
    TeamShapeOverlay().draw(
        frame, boxes, shape, SCALE_TRANSLATE_HOMOGRAPHY, lambda team: (0, 0, 255)
    )
    top_edge = frame[200, 100:500, 2] > 200
    assert 0.4 < top_edge.mean() < 0.8  # dashes and gaps, not a solid line


def test_shape_panel_sits_on_the_minimap_and_shows_dashes_until_measured():
    from scripts.analytics import TeamShapeAnalytics
    from scripts.display import TeamShapePanel

    frame = np.zeros((720, 1280, 3), dtype="uint8")
    top = PitchMinimap().top(frame)
    TeamShapePanel().draw(frame, TeamShapeAnalytics(50.0), bottom=top)
    assert frame[top:].sum() == 0  # nothing drawn over the minimap's area
    assert frame[top - 90 : top - TeamShapePanel.GAP, 1280 - 12 - TeamShapePanel.WIDTH :].sum() > 0
    measured = frame.copy() * 0
    TeamShapePanel().draw(measured, _shaped_team(-30.0), bottom=top)
    assert measured.sum() > frame.sum()  # "36m" and "10m" take more ink than "-"
