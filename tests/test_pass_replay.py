"""Real-footage regression test for ball, possession and passes, fast and
deterministic.

`tests/fixtures/match_5_ball.jsonl.gz` is a realtime recording of what
`PlayerTracker` feeds `BallAnalytics` on match_5.mp4: every worker result's
ball candidates and players' positions on the pitch (metres) and their teams
as the classifier had them at that moment. Team 0 is sky blue (Manchester
City), team 1 white (Tottenham); 2 is keepers/officials.

Hand labels (docs/PLAN.md Plan 3.3), from ball-centred frame strips every
0.08-0.25s: City's pass is cut out by white 8 at ~1.5s; then Tottenham: 8 ->
keeper (3.5s), goal kick -> 15 (15.5s), 15 -> 8 -> 10 (17.5-18.4s), 10 -> 23,
23 -> 10 through two City players' legs (20.5s), 10 -> 2, 2 -> 6 down the
touchline (26.7s), 6 -> 4 (28s), and 4 dribbles alone from 28s to the end.
"""

import gzip
import json
from pathlib import Path

import pytest

from scripts.ball import BallAnalytics

FIXTURE = Path("tests/fixtures/match_5_ball.jsonl.gz")
SKY_BLUE, WHITE = 0, 1


@pytest.fixture(scope="module")
def replayed():
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    analytics = BallAnalytics(fps=50.0)
    known = {}  # like TeamClassifier: every player's latest team, kept after he leaves
    with gzip.open(FIXTURE, "rt") as f:
        for line in f:
            row = json.loads(line)
            known.update({int(k): v for k, v in row["teams"].items() if v is not None})
            analytics.record(
                row["frame_id"],
                [tuple(c) for c in row["candidates"]],
                {int(k): tuple(v) for k, v in row["positions"].items()},
                known.get,
            )
    return analytics


def _events_between(analytics, t0, t1):
    return [(e, team, done) for e, team, done in analytics.passes.counted() if t0 <= e.t <= t1]


def test_city_lose_the_ball_early(replayed):
    interception = _events_between(replayed, 1.0, 2.0)
    assert interception and interception[-1][1:] == (SKY_BLUE, False)
    city = replayed.passes.team_totals()[SKY_BLUE]
    assert city["lost"] == 1 and city["completed"] <= 1  # ball runs past City 3 at 1.1s


def test_back_pass_to_the_keeper_is_completed_not_lost(replayed):
    """The keeper shares OTHER_TEAM with the referee (his own colour)."""
    back_pass = _events_between(replayed, 3.0, 4.0)
    assert [(team, done) for _, team, done in back_pass] == [(WHITE, True)]


def test_goal_kick_is_a_completed_pass(replayed):
    """The keeper steps back for his run-up, so he never has the ball."""
    goal_kick = _events_between(replayed, 15.0, 16.0)
    assert [(e.from_player, team, done) for e, team, done in goal_kick] == [(9, WHITE, True)]


def test_pass_through_opponents_legs_is_one_completed_pass(replayed):
    through = _events_between(replayed, 20.0, 21.0)
    assert [(team, done) for _, team, done in through] == [(WHITE, True)]


def test_pass_to_a_newly_appeared_teammate_is_completed(replayed):
    """The receiver read as sky blue for his first ~2s on screen."""
    touchline_pass = _events_between(replayed, 26.0, 27.0)
    assert [(team, done) for _, team, done in touchline_pass] == [(WHITE, True)]


def test_no_pass_while_one_player_dribbles_to_the_end(replayed):
    """The ball hides behind him; look-alikes 10-24 m away used to take the
    track and book 2-3 passes here."""
    assert _events_between(replayed, 28.5, 36.0) == []


def test_tottenham_move_is_counted(replayed):
    """Labelled: 9-10 completed white passes, none lost."""
    white = replayed.passes.team_totals()[WHITE]
    assert 7 <= white["completed"] <= 10
    assert white["lost"] == 0
