"""Tests for PitchOverlayRenderer -- draws the calibrated pitch outline
(PITCH_LINES, scripts/calibration.py) onto the live video frame.

Fast and deterministic: uses a hand-constructed affine homography (pure
scale + translate, no perspective) so exact expected pixel locations can be
computed and asserted against, rather than just checking "something drew."
"""

import numpy as np

from scripts.calibration import PITCH_LINES
from scripts.display import PitchOverlayRenderer

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
