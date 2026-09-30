"""Overlays: the switches PlayerTracker._draw reads on every frame, set by the
CLI flags before playback and by the web page's checkboxes during it.

The tracker tests draw real frames through PlayerTracker._draw with a stub
model (no YOLO, no video), so an overlay added without its `if` in _draw --
drawn whatever its switch says -- fails here."""

from dataclasses import fields

import numpy as np
import pytest

from scripts.analytics import MatchAnalytics
from scripts.ball import BallAnalytics
from scripts.display import FramePacer, Overlays
from scripts.pipeline import InferenceResult, PlayerTracker

# World (x, y) metres -> pixel (5x + 640, 5y + 360): the whole pitch inside 1280x720.
HOMOGRAPHY = np.array([[5.0, 0.0, 640.0], [0.0, 5.0, 360.0], [0.0, 0.0, 1.0]])


class StubModel:
    def add_callback(self, _event, _callback):
        pass


def test_every_overlay_has_a_label():
    labels = Overlays.labels()
    assert list(labels) == [f.name for f in fields(Overlays)]
    assert all(labels.values())


def test_update_applies_known_overlays():
    overlays = Overlays()
    overlays.update({"markers": True, "minimap": False})
    assert overlays.markers is True
    assert overlays.minimap is False


def test_update_with_an_unknown_name_changes_nothing():
    overlays = Overlays()
    before = overlays.as_dict()
    with pytest.raises(KeyError, match="heatmap"):
        overlays.update({"markers": True, "heatmap": True})
    assert overlays.as_dict() == before


def _tracker(overlays: Overlays) -> PlayerTracker:
    tracker = PlayerTracker("unused.mp4", StubModel(), show_window=False, overlays=overlays)
    tracker.analytics = MatchAnalytics(fps=50.0)
    tracker.ball = BallAnalytics(50.0)
    tracker.homography_history.append((1, HOMOGRAPHY))
    tracker.timeline.add(
        InferenceResult(
            boxes=[(600, 300, 630, 380, 7)],
            frame_id=1,
            captured_at=0.0,
            previous_boxes=[],
            previous_captured_at=None,
            coasting_progress={},
        )
    )
    tracker.fps_overlay.tick(0.0)
    tracker.fps_overlay.tick(0.02)
    return tracker


def _drawn(overlays: Overlays) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    _tracker(overlays)._draw(frame, 1, FramePacer(fps=50.0))
    return frame


def test_all_overlays_off_draws_nothing():
    off = Overlays(**{name: False for name in Overlays.labels()})
    assert _drawn(off).sum() == 0


@pytest.mark.parametrize("name", ["pitch_lines", "markers", "passes", "minimap", "fps"])
def test_each_overlay_on_its_own_draws(name):
    only = Overlays(**{other: other == name for other in Overlays.labels()})
    assert _drawn(only).sum() > 0


def test_a_change_shows_on_the_next_frame():
    overlays = Overlays(**{name: False for name in Overlays.labels()})
    tracker = _tracker(overlays)
    pacer = FramePacer(fps=50.0)
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    tracker._draw(frame, 1, pacer)
    assert frame.sum() == 0
    overlays.update({"minimap": True})
    tracker._draw(frame, 1, pacer)
    assert frame.sum() > 0
