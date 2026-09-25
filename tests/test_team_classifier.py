"""TeamClassifier clustering (docs/PLAN.md Plan 3.1 finding): two kits plus
keepers and officials must come out as two teams and "other", and one kit
split by lighting must not become two teams."""

import numpy as np

from scripts.player import OTHER_TEAM, OTHER_TEAM_NAME, TeamClassifier, color_name

SKY_BLUE = (216, 204, 165)
WHITE = (229, 237, 232)
ORANGE = (84, 130, 228)
DARK = (50, 50, 45)


def _fit(groups):
    """groups: list of (color, n_tracks). Fits once every track has been seen
    (the fit is frozen on first success, so it must not fire on the first
    group alone). Returns the fitted classifier."""
    total = sum(n for _, n in groups)
    classifier = TeamClassifier(
        fit_after_distinct_tracks=total, min_observations_before_fit_eligible=1
    )
    rng = np.random.default_rng(0)
    track = 1
    for color, n in groups:
        for _ in range(n):
            classifier.observe(track, np.array(color, dtype=float) + rng.normal(0, 3, 3))
            track += 1
    assert classifier.centers is not None
    return classifier


def test_two_kits_keepers_and_officials_become_two_teams_and_other():
    classifier = _fit([(SKY_BLUE, 10), (WHITE, 10), (ORANGE, 2), (DARK, 3)])
    blue = classifier.team_for_color(np.array(SKY_BLUE, dtype=float))
    white = classifier.team_for_color(np.array(WHITE, dtype=float))
    assert {blue, white} == {0, 1}
    assert classifier.team_for_color(np.array(ORANGE, dtype=float)) == OTHER_TEAM
    assert classifier.team_for_color(np.array(DARK, dtype=float)) == OTHER_TEAM
    assert classifier.team_color(blue) is not None and classifier.team_color(OTHER_TEAM) is not None


def test_one_kit_under_two_lights_is_still_one_team():
    shaded_white = tuple(c - 22 for c in WHITE)
    classifier = _fit([(SKY_BLUE, 10), (WHITE, 6), (shaded_white, 5), (DARK, 3)])
    assert classifier.team_for_color(np.array(WHITE, dtype=float)) == classifier.team_for_color(
        np.array(shaded_white, dtype=float)
    )
    assert classifier.team_for_color(np.array(SKY_BLUE, dtype=float)) != classifier.team_for_color(
        np.array(WHITE, dtype=float)
    )


def test_jersey_colour_names():
    assert color_name(WHITE) == "white"
    assert color_name(SKY_BLUE) == "sky blue"
    assert color_name(ORANGE) == "orange"
    assert color_name(DARK) == "black"
    assert color_name((83, 215, 206)) == "yellow"  # match_4's fitted yellow kit
    assert color_name((0, 0, 200)) == "red"
    assert color_name((180, 30, 30)) == "blue"


def test_teams_are_named_by_their_kit_colour():
    classifier = _fit([(SKY_BLUE, 10), (WHITE, 10), (ORANGE, 2), (DARK, 3)])
    names = {classifier.team_name(0), classifier.team_name(1)}
    assert names == {"sky blue", "white"}
    assert classifier.team_name(OTHER_TEAM) == OTHER_TEAM_NAME
    assert classifier.team_name(None) == "unknown"


def test_two_kits_reading_the_same_colour_get_distinct_names():
    classifier = _fit([(WHITE, 10), (tuple(c - 45 for c in WHITE), 10), (DARK, 3)])
    assert classifier.team_name(0) != classifier.team_name(1)
