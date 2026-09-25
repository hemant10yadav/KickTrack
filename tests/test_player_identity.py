"""Regression tests for PlayerIdentityManager -- the layer that reconciles
BoT-SORT's own transient track_id into a persistent player_id.

These are synthetic/deterministic (no model, no video) so they run fast and pin
down the exact failure mode found debugging match_4.mp4: a small, low-confidence
box loses its BoT-SORT track for a single frame and reappears under a brand-new
track_id a few pixels away (track 95 -> 159, see docs/PLAN.md Plan 3.5).
"""

import numpy as np

from scripts.player import PlayerIdentityManager


def _box(x1, y1, x2, y2, track_id):
    return (x1, y1, x2, y2, track_id)


def test_same_track_id_stays_same_player():
    """The common case: BoT-SORT keeps reporting one track_id -- no reconciliation
    needed, just a stable mapping."""
    identity = PlayerIdentityManager()
    for frame_id in range(1, 6):
        (result,) = identity.update([_box(100, 100, 130, 160, 5)], frame_id)
        assert result[4] == result[4]  # player_id assigned
    first_player_id = identity.tracker_to_player[5]
    assert len(identity.players) == 1
    assert identity.players[first_player_id].tracker_id == 5


def test_reappearance_under_new_track_id_is_reconciled():
    """The exact 95 -> 159 failure: a tiny box (~12x30px) loses its track for one
    frame and reappears ~9px away under a new track_id. Must be treated as the
    same player, not a new one."""
    identity = PlayerIdentityManager()
    (r1,) = identity.update([_box(1076, 469, 1088, 499, 95)], frame_id=176)
    player_id = r1[4]

    # frame 177: track 95 vanished, track 159 appears ~9px away (matches the
    # real coordinates captured while debugging this failure)
    (r2,) = identity.update([_box(1084, 468, 1097, 499, 159)], frame_id=177)

    assert r2[4] == player_id, "reappearance under a new track_id should keep the same player_id"
    assert len(identity.players) == 1, "must not mint a second player for the same physical target"
    assert identity.switch_log, "the reconciliation should be recorded for instrumentation"


def test_far_away_new_detection_is_not_merged():
    """A genuinely different, distant player must not be merged into an existing
    lost player just because one recently disappeared."""
    identity = PlayerIdentityManager()
    identity.update([_box(100, 100, 130, 160, 1)], frame_id=1)

    # track 1 vanishes; a new track appears far away (a different real person)
    (r,) = identity.update([_box(3000, 1800, 3030, 1860, 2)], frame_id=2)

    assert len(identity.players) == 2, "a distant new detection must get its own player_id"
    assert r[4] != identity.tracker_to_player.get(1, -999)


def test_mismatched_box_size_is_not_merged():
    """A same-position but very differently-sized box shouldn't be merged either
    -- guards against pairing a small player with a large, unrelated detection
    that happens to land nearby (e.g. two different people, one far, one near)."""
    identity = PlayerIdentityManager()
    identity.update([_box(100, 100, 112, 130, 1)], frame_id=1)  # 12x30 (tiny)

    # same neighborhood next frame, but a much bigger box (~4x the height)
    identity.update([_box(95, 95, 145, 215, 2)], frame_id=2)

    assert len(identity.players) == 2, "size-mismatched detections must not be merged"


def test_reappearance_after_max_lost_frames_is_not_merged():
    """Once a player has been gone longer than MAX_LOST_FRAMES, a new nearby
    detection should be treated as a new player, not a stale reconciliation."""
    identity = PlayerIdentityManager()
    identity.update([_box(100, 100, 130, 160, 1)], frame_id=1)

    gap = PlayerIdentityManager.MAX_LOST_FRAMES + 5
    (r,) = identity.update([_box(102, 101, 132, 161, 2)], frame_id=1 + gap)

    # the old player is far enough past MAX_LOST_FRAMES to be retired outright
    # (not merged into) -- the new detection gets a fresh, different player_id
    assert r[4] != 1
    assert not identity.switch_log


def test_untracked_detections_pass_through_unassigned():
    """Boxes with no BoT-SORT track_id (track_id < 0) should pass through with
    player_id -1, not be assigned an identity."""
    identity = PlayerIdentityManager()
    (r,) = identity.update([_box(10, 10, 20, 20, -1)], frame_id=1)
    assert r[4] == -1
    assert len(identity.players) == 0


def test_lost_player_is_eventually_retired():
    """A player that never reappears should be forgotten (not accumulate forever)."""
    identity = PlayerIdentityManager()
    identity.update([_box(100, 100, 130, 160, 1)], frame_id=1)
    assert len(identity.players) == 1

    # advance far past the lost-frame budget with no matching detections
    identity.update([], frame_id=1 + PlayerIdentityManager.MAX_LOST_FRAMES + 1)
    assert len(identity.players) == 0


# --- Occlusion / ID-swap handling (docs/PLAN.md Plan 3.6) --------------------
#
# The tests above cover *fragmentation*: one player, new track_id after a gap.
# The tests below cover the other failure mode -- two players converging, where
# the risk is not losing a player but silently swapping two of them, which
# would quietly attribute one player's distance/heatmap to the other.


class FakeSampler:
    """Stand-in for the real JerseySampler: a fixed raw jersey color per box x-center."""

    def __init__(self, color_by_box_center=None):
        self.color_by_box_center = color_by_box_center or {}

    def color_of(self, bbox):
        color = self.color_by_box_center.get((bbox[0] + bbox[2]) // 2)
        return None if color is None else np.array(color, dtype=float)


WHITE = (230, 238, 238)
BLUE = (216, 204, 165)  # match_5's sky-blue kit, ~75 BGR units from white
ORANGE = (84, 130, 228)  # match_5's goalkeeper, ~190 from white


def test_crossing_players_keep_their_identities():
    """Two players converge until the detector returns a single merged box, then
    separate under brand-new track_ids. Each must come out with the identity it
    went in with -- resolved from the motion each had *before* the merge, since
    the merged box's own apparent motion belongs to neither of them."""
    identity = PlayerIdentityManager()

    # Approach: A moves right (+20px/cycle), B moves left (-20px/cycle), same row,
    # until they are almost touching (A at 180-210, B at 220-250).
    for step, frame_id in enumerate(range(1, 6)):
        ax = 100 + step * 20
        bx = 300 - step * 20
        identity.update([_box(ax, 100, ax + 30, 160, 1), _box(bx, 100, bx + 30, 160, 2)], frame_id)
    player_a = identity.tracker_to_player[1]
    player_b = identity.tracker_to_player[2]
    assert player_a != player_b

    # Merge: the detector returns one box spanning both, under a single id.
    for frame_id in (6, 7):
        identity.update([_box(180, 100, 250, 160, 1)], frame_id)

    # Separation: they emerge on the far sides of each other, under new ids.
    # A (was moving right) is the right-hand box; B is the left-hand box.
    results = identity.update(
        [_box(250, 100, 280, 160, 7), _box(160, 100, 190, 160, 8)], frame_id=8
    )
    by_center = {(r[0] + r[2]) // 2: r[4] for r in results}

    assert by_center[265] == player_a, "the right-hand box is the player who was moving right"
    assert by_center[175] == player_b, "the left-hand box is the player who was moving left"


def test_another_players_jersey_is_never_merged():
    """A lost player and a nearby new detection wearing a jersey that clearly
    belongs to some *other* known player cannot be the same person -- the
    merge must be refused outright."""
    identity = PlayerIdentityManager()
    sampler = FakeSampler({115: BLUE, 515: WHITE})
    identity.update([_box(100, 100, 130, 160, 1), _box(500, 100, 530, 160, 2)], 1, sampler)

    # track 1 (blue) vanishes; a white box appears exactly where it was
    sampler = FakeSampler({115: WHITE, 515: WHITE})
    results = identity.update(
        [_box(100, 100, 130, 160, 3), _box(500, 100, 530, 160, 2)], 2, sampler
    )

    assert len(identity.players) == 3, "a cross-jersey merge must mint a new player instead"
    assert results[0][4] != 1
    assert identity.jersey_blocked_merges >= 1


def test_same_jersey_nearby_detection_is_still_merged():
    """Control for the test above: identical geometry, same jersey -> merged."""
    identity = PlayerIdentityManager()
    sampler = FakeSampler({115: BLUE, 515: WHITE})
    identity.update([_box(100, 100, 130, 160, 1), _box(500, 100, 530, 160, 2)], 1, sampler)

    results = identity.update(
        [_box(100, 100, 130, 160, 3), _box(500, 100, 530, 160, 2)], 2, sampler
    )

    assert len(identity.players) == 2
    assert results[0][4] == 1


def test_odd_lighting_alone_does_not_block_a_merge():
    """A color far from the player's own jersey but resembling nobody else's is
    noise (shadow, crowd behind the crop), not evidence of another person --
    match_4-style footage produces this constantly and must still reconcile."""
    identity = PlayerIdentityManager()
    sampler = FakeSampler({115: BLUE, 515: WHITE})
    identity.update([_box(100, 100, 130, 160, 1), _box(500, 100, 530, 160, 2)], 1, sampler)

    sampler = FakeSampler({115: (120, 120, 120), 515: WHITE})  # grey: like no kit here
    results = identity.update(
        [_box(100, 100, 130, 160, 3), _box(500, 100, 530, 160, 2)], 2, sampler
    )

    assert results[0][4] == 1
    assert len(identity.players) == 2


def test_reappearances_are_assigned_jointly_not_greedily():
    """Two players reappear in the same cycle. Taking each detection's own best
    match in turn gives the first detection the player that the second one is a
    far better fit for; the assignment has to be solved across all of them at
    once."""
    identity = PlayerIdentityManager()
    identity.update([_box(85, 100, 115, 160, 1)], frame_id=1)  # player A, center 100
    identity.update([_box(235, 100, 265, 160, 2)], frame_id=1)  # player B, center 250
    player_a = identity.tracker_to_player[1]
    player_b = identity.tracker_to_player[2]

    # Both vanish, then reappear under new ids. Detection order matters to a
    # greedy matcher: det@160 is nearer A (60px) than B (90px) and would claim
    # A first, leaving det@100 -- a perfect match for A -- stuck with B.
    results = identity.update(
        [_box(145, 100, 175, 160, 9), _box(85, 100, 115, 160, 10)], frame_id=2
    )
    by_center = {(r[0] + r[2]) // 2: r[4] for r in results}

    assert by_center[100] == player_a, "the exact-position detection must win player A"
    assert by_center[160] == player_b


def test_velocity_is_frozen_while_occluded():
    """While a player's box is merged with another's, its apparent motion is an
    artifact of the merge, not of the player. Velocity must hold at its last
    clean value so post-occlusion prediction is still meaningful."""
    identity = PlayerIdentityManager()
    for step, frame_id in enumerate(range(1, 4)):
        ax = 100 + step * 10
        identity.update([_box(ax, 100, ax + 30, 160, 1), _box(500, 100, 530, 160, 2)], frame_id)
    player_a = identity.tracker_to_player[1]
    clean_vx = identity.players[player_a].vx
    assert clean_vx == 10

    # track 1's box now overlaps track 2's -- both are "occluded"
    identity.update([_box(400, 100, 500, 160, 1), _box(480, 100, 510, 160, 2)], frame_id=4)

    assert identity.players[player_a].vx == clean_vx, "velocity must not absorb the merge jump"
    assert identity.players[player_a].occluded


def test_occluded_players_are_reported_for_analytics():
    """Positions sampled during an occlusion are unreliable, so downstream stats
    need to know which players those are and drop those samples."""
    identity = PlayerIdentityManager()
    identity.update([_box(100, 100, 130, 160, 1), _box(500, 100, 530, 160, 2)], frame_id=1)
    assert identity.occluded_player_ids == set()

    identity.update([_box(100, 100, 130, 160, 1), _box(110, 100, 140, 160, 2)], frame_id=2)

    assert identity.occluded_player_ids == {
        identity.tracker_to_player[1],
        identity.tracker_to_player[2],
    }
