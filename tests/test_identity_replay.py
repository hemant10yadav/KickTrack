"""Real-footage regression test for identity swaps, fast and deterministic.

`tests/fixtures/match_5_tracks.jsonl.gz` is a drop-free recording of what the
pipeline feeds PlayerIdentityManager on match_5.mp4: every frame's raw BoT-SORT
boxes (after `extract_boxes`) plus the jersey color `extract_jersey_color`
sampled from each one. Replaying it exercises the split suppressor, team
classifier and identity manager exactly as `IdentityResolver` does, without YOLO
or the video, so the two body swaps found on this clip (docs/PLAN.md Plan 2.8)
can be pinned to hand-checked frames:

* the goalkeeper's track walked off with a defender who had stood in front of
  him (frame 596), and the goalkeeper got a fresh track four frames later;
* a blue player's track walked off with a white one after they ran together
  (frame 1629), and the blue player got a fresh track four frames later.

Neither produces a lost track, so nothing before Plan 2.8 could see them.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.player import (
    PlayerIdentityManager,
    SplitDetectionSuppressor,
    TeamClassifier,
    boxes_overlap,
)

FIXTURE = Path("tests/fixtures/match_5_tracks.jsonl.gz")

# (frame, bottom-centre x, bottom y) of a hand-checked body, from the dump.
PROBES = {
    "goalkeeper@300": (300, 1427, 537),
    "goalkeeper@650": (650, 1635, 436),
    "goalkeeper@1000": (1000, 1846, 456),
    "blue@1600": (1600, 659, 648),
    "white@1600": (1600, 678, 620),
    "blue@1660": (1660, 699, 634),
    "white@1660": (1660, 721, 634),
    # a third pair: their tracks exchanged bodies while both stayed alive (f172),
    # then the detector merged them into one box for ~70 frames (f1668-1738).
    "white@165": (165, 1030, 377),
    "blue@1748": (1748, 1163, 460),
    "white@1748": (1748, 1181, 459),
}


class RecordedSampler:
    """JerseySampler stand-in serving the colors recorded for this frame's boxes."""

    def __init__(self, colors):
        self.colors = colors

    def color_of(self, bbox):
        color = self.colors.get(tuple(bbox))
        return None if color is None else np.array(color, dtype=float)


def replay(path: Path):
    classifier = TeamClassifier()
    identity = PlayerIdentityManager()
    splits = SplitDetectionSuppressor()
    owners = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            frame_id = record["f"]
            colors = {tuple(b[:4]): b[5] for b in record["b"]}
            sampler = RecordedSampler(colors)
            boxes = splits.update([tuple(b[:5]) for b in record["b"]], frame_id)
            boxes = identity.update(boxes, frame_id, sampler)
            for i, (x1, y1, x2, y2, player_id) in enumerate(boxes):
                if player_id < 0:
                    continue
                overlapped = any(
                    boxes_overlap((x1, y1, x2, y2), other[:4])
                    for j, other in enumerate(boxes)
                    if j != i
                )
                if not overlapped:
                    classifier.observe(player_id, sampler.color_of((x1, y1, x2, y2)))
            for name, (probe_frame, px, py) in PROBES.items():
                if probe_frame == frame_id:
                    nearest = min(boxes, key=lambda b: abs((b[0] + b[2]) / 2 - px) + abs(b[3] - py))
                    owners[name] = nearest[4]
    return identity, owners


@pytest.fixture(scope="module")
def replayed():
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found: {FIXTURE}")
    return replay(FIXTURE)


def test_goalkeeper_keeps_his_identity_after_a_defender_walks_through(replayed):
    _, owners = replayed
    assert owners["goalkeeper@300"] == owners["goalkeeper@650"] == owners["goalkeeper@1000"]


def test_players_who_ran_together_come_apart_with_their_own_identities(replayed):
    _, owners = replayed
    assert owners["blue@1600"] != owners["white@1600"]
    assert owners["blue@1660"] == owners["blue@1600"]
    assert owners["white@1660"] == owners["white@1600"]


def test_tracks_that_exchanged_bodies_are_exchanged_back(replayed):
    _, owners = replayed
    assert owners["white@1748"] == owners["white@165"]
    assert owners["blue@1748"] != owners["white@1748"]


def test_identity_churn_ceilings(replayed):
    """Loose ceilings so a regression in the merge/transfer gates shows up as a
    number, not just as a wrong body: measured 52 ids / 13 corrections on this
    fixture when Plan 2.8 landed (53 ids before it)."""
    identity, _ = replayed
    assert identity.total_players_minted <= 62
    assert len(identity.swap_log) <= 20
