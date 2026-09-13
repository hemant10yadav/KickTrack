"""Regression tests for PlayerIdentityManager -- the layer that reconciles
BoT-SORT's own transient track_id into a persistent player_id.

These are synthetic/deterministic (no model, no video) so they run fast and pin
down the exact failure mode found debugging match_4.mp4: a small, low-confidence
box loses its BoT-SORT track for a single frame and reappears under a brand-new
track_id a few pixels away (track 95 -> 159, see docs/PLAN.md Plan 3.5).
"""

from scripts.player import PlayerIdentityManager


def _box(x1, y1, x2, y2, track_id):
    return (x1, y1, x2, y2, track_id)


class _FakeClassifier:
    """Minimal stand-in for TeamClassifier: just the two bits of state
    PlayerIdentityManager reads (track_colors keyed by player_id, and a
    controllable team_for lookup) -- no kmeans fitting needed for these tests.
    """

    def __init__(self):
        self.track_colors: dict[int, tuple] = {}
        self.teams: dict[int, int] = {}

    def team_for(self, player_id):
        return self.teams.get(player_id)


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


def test_long_gap_reappearance_is_merged_by_jersey_color():
    """Beyond MAX_LOST_FRAMES, position prediction is unreliable, but a distinctive
    jersey color reappearing within LONG_LOST_MAX_FRAMES should still be reconciled
    to the same player_id instead of minting a new one."""
    classifier = _FakeClassifier()
    identity = PlayerIdentityManager(classifier=classifier)

    identity.update([_box(100, 100, 130, 160, 1)], frame_id=1)
    original_player_id = identity.tracker_to_player[1]
    classifier.track_colors[original_player_id] = (10, 20, 200)  # e.g. a red jersey

    # player vanishes for longer than MAX_LOST_FRAMES but within LONG_LOST_MAX_FRAMES,
    # reappearing far from where velocity extrapolation would predict
    gap = PlayerIdentityManager.MAX_LOST_FRAMES + 10
    reappear_frame = 1 + gap
    (r,) = identity.update([_box(900, 700, 930, 760, 2)], frame_id=reappear_frame)
    new_player_id = r[4]
    assert new_player_id != original_player_id, "no color sample yet -- must mint provisionally"

    # jersey color for the new track_id is now observed (as pipeline.py would do,
    # keyed by the just-minted player_id) and closely matches the original
    classifier.track_colors[new_player_id] = (12, 18, 195)

    (r2,) = identity.update([_box(902, 702, 932, 762, 2)], frame_id=reappear_frame + 1)
    assert r2[4] == original_player_id, (
        "matching jersey color should merge into the long-lost player"
    )
    assert len(identity.players) == 1


def test_long_gap_reappearance_with_different_color_is_not_merged():
    """A same-timing, same-position reappearance with a clearly different jersey
    color must not be merged -- it's a different player, not the same one."""
    classifier = _FakeClassifier()
    identity = PlayerIdentityManager(classifier=classifier)

    identity.update([_box(100, 100, 130, 160, 1)], frame_id=1)
    original_player_id = identity.tracker_to_player[1]
    classifier.track_colors[original_player_id] = (10, 20, 200)  # red jersey

    gap = PlayerIdentityManager.MAX_LOST_FRAMES + 10
    reappear_frame = 1 + gap
    (r,) = identity.update([_box(102, 101, 132, 161, 2)], frame_id=reappear_frame)
    new_player_id = r[4]
    classifier.track_colors[new_player_id] = (200, 180, 15)  # clearly different (blue) jersey

    (r2,) = identity.update([_box(103, 102, 133, 162, 2)], frame_id=reappear_frame + 1)
    assert r2[4] == new_player_id, "mismatched jersey color must not be merged"
    assert len(identity.players) == 2


def test_crossing_swap_is_detected_and_corrected():
    """Two continuously-active track_ids never go through _reconcile at all, so a
    mid-crossing swap has to be caught a different way: a sustained, mutual
    team-classification mismatch between two nearby players."""
    classifier = _FakeClassifier()
    identity = PlayerIdentityManager(classifier=classifier)

    box_a = _box(100, 100, 130, 160, 1)  # player A, track 1
    box_b = _box(140, 100, 170, 160, 2)  # player B, track 2, close by

    classifier.teams = {}
    (ra, rb) = identity.update([box_a, box_b], frame_id=1)
    player_a, player_b = ra[4], rb[4]
    classifier.teams = {player_a: 0, player_b: 1}
    identity.update([box_a, box_b], frame_id=2)  # locks in each player's `team`

    # crossing happens: BoT-SORT now reports track 1's detections as classified
    # into team 1 (player B's team) and track 2's as team 0 (player A's team),
    # for several consecutive cycles, while the boxes stay close together
    classifier.teams = {player_a: 1, player_b: 0}
    for frame_id in range(3, 3 + PlayerIdentityManager.SWAP_MISMATCH_CYCLES):
        identity.update([box_a, box_b], frame_id=frame_id)

    assert identity.swap_log, (
        "a sustained mutual team mismatch between nearby players should be corrected"
    )
    # tracker mapping should now be swapped: track 1 -> player_b, track 2 -> player_a
    assert identity.tracker_to_player[1] == player_b
    assert identity.tracker_to_player[2] == player_a


def test_distant_mismatched_players_are_not_swapped():
    """A sustained team mismatch alone isn't enough -- if the two players aren't
    actually near each other, it can't be a crossing, so don't touch the mapping."""
    classifier = _FakeClassifier()
    identity = PlayerIdentityManager(classifier=classifier)

    box_a = _box(100, 100, 130, 160, 1)
    box_b = _box(3000, 1800, 3030, 1860, 2)  # far away

    (ra, rb) = identity.update([box_a, box_b], frame_id=1)
    player_a, player_b = ra[4], rb[4]
    classifier.teams = {player_a: 0, player_b: 1}
    identity.update([box_a, box_b], frame_id=2)

    classifier.teams = {player_a: 1, player_b: 0}
    for frame_id in range(3, 3 + PlayerIdentityManager.SWAP_MISMATCH_CYCLES):
        identity.update([box_a, box_b], frame_id=frame_id)

    assert not identity.swap_log, "distant players must not be swapped just for a team mismatch"
    assert identity.tracker_to_player[1] == player_a
    assert identity.tracker_to_player[2] == player_b
