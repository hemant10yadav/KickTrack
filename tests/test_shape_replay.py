"""Real-footage regression test for team shape, fast and deterministic.

`tests/fixtures/match_5_shape.jsonl.gz` is a realtime recording of what
`PlayerTracker` feeds `TeamShapeAnalytics` on match_5.mp4: every worker
result's players on the pitch (metres), their teams as the classifier read
them at that moment, the pitch -> frame homography, the working frame size
and the side in possession. Team 0 is sky blue (Manchester City), team 1
white (Tottenham).

Checked by eye on the video (docs/PLAN.md Plan 3.4), with each team's outline
and line drawn on the frame: City press with their last man just past
halfway from ~11 s to ~24 s while Tottenham build from the back, and are
camped around their own box by 31-34 s; Tottenham's keeper is at +x, City's
at -x. From 27.6 s to 29.8 s the far-side assistant referee, read as sky
blue, runs the touchline 36 m from the nearest City player.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analytics import TeamShapeAnalytics
from scripts.ball import TeamHistory

FIXTURE = Path("tests/fixtures/match_5_shape.jsonl.gz")
FPS = 50.0
CITY, SPURS = 0, 1


@pytest.fixture(scope="module")
def replayed():
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    shape = TeamShapeAnalytics(fps=FPS)
    known = {}  # like TeamClassifier: every player's latest team, kept after he leaves
    history = TeamHistory()  # like BallAnalytics.team_now, which the pipeline passes
    samples = []
    with gzip.open(FIXTURE, "rt") as f:
        for line in f:
            row = json.loads(line)
            known.update({int(k): v for k, v in row["teams"].items() if v is not None})
            t = row["frame_id"] / FPS
            positions = {int(k): tuple(v) for k, v in row["positions"].items()}
            history.record(t, positions, known.get)
            homography = None if row["H"] is None else np.array(row["H"])
            sampled = shape.results
            shape.record(
                row["frame_id"],
                positions,
                lambda pid, t=t: history.team_at(pid, t),
                homography,
                tuple(row["size"]),
                row["possession"],
            )
            if shape.results > sampled:  # a sample was taken (every SAMPLE_S)
                samples.extend((t, team, s) for team, s in shape.latest.items())
    return shape, samples


def _values(samples, team, metric, t0, t1):
    return [
        getattr(s, metric)
        for t, tm, s in samples
        if tm == team and t0 <= t <= t1 and getattr(s, metric) is not None
    ]


def test_each_team_defends_its_keepers_goal(replayed):
    shape, _ = replayed
    assert shape.own_goal == {CITY: -1, SPURS: 1}


def test_city_press_high_then_defend_deep(replayed):
    _, samples = replayed
    pressing = _values(samples, CITY, "line_m", 11.0, 24.0)
    deep = _values(samples, CITY, "line_m", 31.0, 34.0)
    assert len(pressing) > 80 and np.median(pressing) > 48  # last man around halfway (52.5 m)
    assert len(deep) > 20 and np.median(deep) < 20  # at the edge of their box (16.5 m)


def test_linesman_does_not_stretch_city(replayed):
    _, samples = replayed
    widths = _values(samples, CITY, "width_m", 27.6, 29.8)
    assert widths and max(widths) < 45  # 68 m with him in


def test_city_defend_narrower_than_spurs_attack(replayed):
    """Tottenham keep the ball from 1.5 s: City's block without it is
    narrower than Tottenham spread with it."""
    shape, _ = replayed
    city = shape.team_stats(CITY)["width_m"]["out_of_possession"]
    spurs = shape.team_stats(SPURS)["width_m"]["in_possession"]
    assert city is not None and spurs is not None and city < spurs - 10


def _defence_sizes(samples, team, t0, t1):
    return [len(s.lines[0]) for t, tm, s in samples if tm == team and t0 <= t <= t1 and s.lines]


def test_city_lines_press_with_few_back_then_drop_into_a_block(replayed):
    """Pressing (16-22 s) City keep two or three back; camped in their box
    (26-34 s) four or more make the defence line."""
    _, samples = replayed
    pressing = _defence_sizes(samples, CITY, 16.0, 22.0)
    block = _defence_sizes(samples, CITY, 26.0, 34.0)
    assert len(pressing) > 40 and np.median(pressing) <= 3
    assert len(block) > 40 and min(block) >= 4


def test_spurs_back_four_when_they_come_into_view(replayed):
    _, samples = replayed
    labels = [
        s.formation for t, tm, s in samples if tm == SPURS and 30.5 <= t <= 34.5 and s.formation
    ]
    assert labels and all(label.startswith("4-") for label in labels)


def test_spurs_depth_only_when_both_ends_are_in_view(replayed):
    """While City press, Tottenham stretch from their box to halfway and one
    end is always out of the picture: no depth is measured then."""
    _, samples = replayed
    assert not _values(samples, SPURS, "depth_m", 11.0, 20.0)
