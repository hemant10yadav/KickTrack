"""Regression tests for SplitDetectionSuppressor -- the filter that collapses two
boxes on one body back into one detection.

The case (docs/PLAN.md Plan 2.7, found on match_5.mp4 f1443): YOLO emits both a
box around a running player including their extended leg, and a second box around
just the torso. BoT-SORT gives the second one its own track_id, so one player
arrives downstream as two players and gets counted twice.
"""

from scripts.player import SplitDetectionSuppressor


def _box(x1, y1, x2, y2, track_id):
    return (x1, y1, x2, y2, track_id)


def test_split_box_on_one_body_is_suppressed():
    """The real f1443 geometry: identical top and bottom edge, one box horizontally
    inside the other. Two people cannot share a head line and a feet line to the
    pixel, so this is one player detected twice."""
    s = SplitDetectionSuppressor()
    s.update([_box(1221, 664, 1242, 715, 5)], frame_id=1)  # established track

    kept = s.update([_box(1221, 664, 1242, 715, 5), _box(1221, 664, 1256, 715, 57)], frame_id=2)

    assert [b[4] for b in kept] == [5], "the newer track is the split, and must be dropped"


def test_established_larger_box_wins_over_newer_smaller_one():
    """Which box is bigger says nothing about which is real -- measured on
    match_5, the established track was the smaller box in 42 of 114 split pairs.
    Age decides, not size."""
    s = SplitDetectionSuppressor()
    s.update([_box(1221, 664, 1256, 715, 57)], frame_id=1)  # established: the LARGER

    kept = s.update([_box(1221, 664, 1242, 715, 5), _box(1221, 664, 1256, 715, 57)], frame_id=2)

    assert [b[4] for b in kept] == [57]


def test_occluded_player_behind_another_is_kept():
    """Same feet line but a lower head: a shorter player standing behind a taller
    one, mostly hidden. Contained, but a real second person -- must survive."""
    s = SplitDetectionSuppressor()
    kept = s.update([_box(100, 100, 130, 200, 1), _box(104, 150, 126, 200, 2)], frame_id=1)

    assert sorted(b[4] for b in kept) == [1, 2]


def test_two_separate_players_are_both_kept():
    s = SplitDetectionSuppressor()
    kept = s.update([_box(100, 100, 130, 160, 1), _box(400, 100, 430, 160, 2)], frame_id=1)
    assert sorted(b[4] for b in kept) == [1, 2]


def test_partially_overlapping_players_are_both_kept():
    """Two players contesting a ball overlap a lot, but neither box contains the
    other -- that is an occlusion to be handled downstream, not a split."""
    s = SplitDetectionSuppressor()
    kept = s.update([_box(100, 100, 140, 160, 1), _box(125, 100, 165, 160, 2)], frame_id=1)
    assert sorted(b[4] for b in kept) == [1, 2]


def test_untracked_detections_pass_through():
    s = SplitDetectionSuppressor()
    kept = s.update([_box(10, 10, 20, 40, -1), _box(10, 10, 24, 40, -1)], frame_id=1)
    assert len(kept) == 2


def test_suppression_is_counted_for_instrumentation():
    s = SplitDetectionSuppressor()
    s.update([_box(1221, 664, 1242, 715, 5)], frame_id=1)
    s.update([_box(1221, 664, 1242, 715, 5), _box(1221, 664, 1256, 715, 57)], frame_id=2)
    assert s.suppressed_detections == 1
    assert 57 in s.suppressed_tracks


def test_phantom_track_resolves_to_the_real_one_when_seen_alone():
    """On later cycles the split box can be the only detection on that player.
    It must still resolve to the established track, or it becomes a second
    player -- deleting the box instead was measured to cause exactly that."""
    s = SplitDetectionSuppressor()
    s.update([_box(1221, 664, 1242, 715, 5)], frame_id=1)
    s.update([_box(1221, 664, 1242, 715, 5), _box(1221, 664, 1256, 715, 57)], frame_id=2)

    (kept,) = s.update([_box(1221, 664, 1256, 715, 57)], frame_id=3)

    assert kept[4] == 5, "the phantom track must report as the track it is part of"


def test_alias_is_revoked_when_the_boxes_come_apart():
    """If the two tracks are later seen apart, they were two real detections --
    the alias must not outlive the geometry that justified it."""
    s = SplitDetectionSuppressor()
    s.update([_box(1221, 664, 1242, 715, 5)], frame_id=1)
    s.update([_box(1221, 664, 1242, 715, 5), _box(1221, 664, 1256, 715, 57)], frame_id=2)

    kept = s.update([_box(1221, 664, 1242, 715, 5), _box(1500, 664, 1535, 715, 57)], frame_id=3)

    assert sorted(b[4] for b in kept) == [5, 57]


# --- Containers (docs/PLAN.md Plan 2.9) ----------------------------------------
#
# YOLO26 returns, for two overlapping players, a box for each *and* a box around
# both. The union is not a third player.


def test_container_around_two_bodies_is_dropped():
    """The real f588 geometry on match_5 with yolo26s: keeper box, defender box
    below him, and one tall box spanning both."""
    s = SplitDetectionSuppressor()
    kept = s.update(
        [
            _box(1614, 390, 1643, 486, 9),  # container: keeper + defender
            _box(1616, 392, 1642, 440, 40),  # keeper
            _box(1617, 438, 1643, 487, 41),  # defender
        ],
        frame_id=1,
    )

    assert sorted(b[4] for b in kept) == [40, 41]
    assert s.suppressed_containers == 1


def test_one_player_behind_another_is_not_a_container():
    """A box holding a single shorter box is a player occluded behind another --
    the Plan 2.7 case that must survive -- not a union of two."""
    s = SplitDetectionSuppressor()
    kept = s.update([_box(100, 100, 130, 200, 1), _box(104, 150, 126, 200, 2)], frame_id=1)
    assert sorted(b[4] for b in kept) == [1, 2]
    assert s.suppressed_containers == 0


def test_container_is_dropped_not_aliased():
    """Dropping, not aliasing: the container belongs to neither body, so its
    track must not be rewritten onto one of them."""
    s = SplitDetectionSuppressor()
    s.update([_box(1614, 390, 1643, 486, 9)], frame_id=1)
    s.update(
        [
            _box(1614, 390, 1643, 486, 9),
            _box(1616, 392, 1642, 440, 40),
            _box(1617, 438, 1643, 487, 41),
        ],
        frame_id=2,
    )
    assert 9 not in s.alias and 40 not in s.alias and 41 not in s.alias
