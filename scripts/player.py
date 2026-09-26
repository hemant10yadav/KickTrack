from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Colour groups on a pitch: two kits, the keepers, the officials. Fitting only
# three (the original "2 teams + other") merged the two *kits* on match_5, where
# the orange keeper and stewards formed the third group and sky blue / white were
# the closest pair -- every outfield marker went grey (docs/PLAN.md Plan 3.1).
NUM_TEAM_CLUSTERS = 4
MIN_KIT_SEPARATION = 40.0  # clusters closer than this (BGR) are one kit under two lights
OTHER_TEAM = 2  # team id for everyone who is not in one of the two biggest groups
OTHER_TEAM_NAME = "others"


def color_name(bgr) -> str:
    """A plain jersey colour name for a BGR value, so a team can be called
    "white" or "yellow" on screen instead of a cluster number. Decided in HSV:
    little saturation is white / grey / black by brightness, otherwise the hue
    band; a light, weakly saturated blue is "sky blue" (match_5's kit)."""
    b, g, r = (float(v) for v in bgr)
    hsv = cv2.cvtColor(np.uint8([[[int(b), int(g), int(r)]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat, val = int(hsv[0]) * 2, int(hsv[1]) / 255, int(hsv[2]) / 255  # hue in degrees
    if sat < 0.18:
        if val > 0.75:
            return "white"
        return "black" if val < 0.3 else "grey"
    if sat < 0.35 and val > 0.7 and 175 <= hue <= 260:
        return "sky blue"
    if hue < 15 or hue >= 340:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 205:
        return "sky blue" if val > 0.7 else "teal"
    if hue < 260:
        return "blue"
    if hue < 300:
        return "purple"
    return "pink"


GRASS_HSV_LOW, GRASS_HSV_HIGH = (35, 40, 40), (85, 255, 255)  # OpenCV HSV range of pitch green
# Distinct tracks (each with a few samples) seen before the team clusters are
# fit and frozen. Was 25: with the yolov8 tracker that arrived within seconds
# because it minted phantom ids freely; YOLO26 + the identity layer (Plan 2.8/2.9)
# mint ~40 ids in a whole 35s clip, so 25 distinct tracks came late enough that
# match_5 was still uncoloured 24s in. Two teams plus officials is ~23 people;
# 16 is reached in the first seconds and still spans both kits.
TEAM_FIT_AFTER_SAMPLES = 16


def extract_jersey_color(frame, x1, y1, x2, y2):
    """Sample the dominant jersey color from a player's bounding box.

    Crops to the torso (avoids head/legs/background at the box edges) and masks
    out grass green so the pitch behind the player doesn't skew the color.
    """
    h = y2 - y1
    w = x2 - x1
    ty1 = max(0, y1 + int(h * 0.15))
    ty2 = max(0, y1 + int(h * 0.55))
    tx1 = max(0, x1 + int(w * 0.2))
    tx2 = max(0, x1 + int(w * 0.8))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GRASS_HSV_LOW, GRASS_HSV_HIGH)
    non_green = crop[green_mask == 0]
    pixels = non_green if len(non_green) >= 10 else crop.reshape(-1, 3)
    # Lower median via partition: half the cost of np.median on these tiny
    # crops, and this runs for every box on every worker cycle.
    middle = len(pixels) // 2
    color = np.partition(pixels, middle, axis=0)[middle].astype(float)
    # A median that is itself pitch green means the crop missed the body (a
    # wide box around a running player, a shadowed yellow kit half-masked as
    # grass): that is no jersey at all, and must not be mistaken for one.
    return None if is_grass_color(color) else color


def is_grass_color(color) -> bool:
    hsv = cv2.cvtColor(np.uint8([[np.clip(color, 0, 255)]]), cv2.COLOR_BGR2HSV)[0, 0]
    return bool(np.all(hsv >= GRASS_HSV_LOW) and np.all(hsv <= GRASS_HSV_HIGH))


def boxes_overlap(box_a, box_b, overlap_ratio_threshold=0.2, iou_threshold=0.15) -> bool:
    """True if two boxes overlap enough that jersey-color sampling would risk
    picking up the other player — e.g. two opposing players contesting a ball.
    Checked both as a fraction of the smaller box's area (catches a small box mostly
    swallowed by a bigger one) and as IoU (catches two similarly-sized boxes
    overlapping less than fully).
    """
    intersection = _intersection(box_a, box_b)
    if intersection == 0:
        return False

    area_a, area_b = _area(box_a), _area(box_b)
    overlap_ratio = intersection / min(area_a, area_b)
    iou = intersection / (area_a + area_b - intersection)
    return overlap_ratio > overlap_ratio_threshold or iou > iou_threshold


def pairwise_overlap(boxes_a, boxes_b, overlap_ratio_threshold=0.2, iou_threshold=0.15):
    """boxes_overlap for every pair at once: (len(a), len(b)) bool matrix from
    two (n, 4) xyxy arrays. Same test, vectorized -- the per-cycle occlusion
    bookkeeping asks it ~700 times, which in Python was the single largest cost
    of the identity layer."""
    a, b = np.asarray(boxes_a, dtype=float), np.asarray(boxes_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=bool)
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    intersection = ix * iy
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    smaller = np.maximum(1, np.minimum(area_a[:, None], area_b[None, :]))
    union = np.maximum(1, area_a[:, None] + area_b[None, :] - intersection)
    return (intersection > 0) & (
        (intersection / smaller > overlap_ratio_threshold) | (intersection / union > iou_threshold)
    )


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(box_a, box_b) -> int:
    iw = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    ih = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return iw * ih


class SplitDetectionSuppressor:
    """Collapses two boxes on one body back into a single detection.

    YOLO sometimes returns both a box around a whole running player (wide, because
    an extended leg stretches it) and a second box around just their torso. They
    are separate detections, so BoT-SORT gives the second one its own track_id and
    one player arrives downstream as two -- two markers, two team votes, and in
    Plan 3 two sets of distance/heatmap numbers for one person.

    A split is recognised geometrically: one box horizontally inside the other
    while both share a top AND a bottom edge. Two different people cannot share a
    head line and a feet line to within a few percent of their height, so this
    does not fire on a genuinely occluded player standing behind another -- those
    share a feet line but not a head line (measured on match_5: 114 pairs share
    both edges, ~106 share only the feet line, and the latter are left alone).

    The phantom track is **aliased onto the real one, not deleted**. Simply
    dropping its box was measured to be much worse than the problem: the track
    vanishes, PlayerIdentityManager marks that player lost, and the next cycle it
    reappears unmapped and gets reconciled -- on match_5 that took reconciliations
    from 1 to 28 and reintroduced 5 identity swaps. Rewriting the phantom's
    track_id to the surviving one instead keeps a single stable track, so the
    split contributes no identity at all, and the alias still resolves correctly
    on the later cycles where the phantom is the only box on that player.

    Which track survives is decided by age, not size. Size is not evidence: the
    established track was the *smaller* box in 42 of those 114 pairs.

    The opposite phantom is a **container**: one box around two overlapping
    players *alongside* a box for each of them (YOLO26 returns all three on
    ~2% of match_5 frames, v8s rarely). Two bodies plus their union is three
    detections for two people. The container is dropped outright rather than
    aliased: it belongs to neither body, and whichever track it carried is
    reconciled onto the right body by PlayerIdentityManager from that body's
    own box (position + jersey), which is exactly the case that layer exists for.
    """

    CONTAINMENT = 0.9  # intersection as a fraction of the smaller box
    EDGE_TOLERANCE = 0.05  # top/bottom agreement, as a fraction of the smaller box's height
    HELD_INSIDE = 0.7  # a body counts as held by a container when this much of it is inside
    HELD_MAX_HEIGHT = 0.85  # ...and it is clearly shorter than the container (not a split)

    def __init__(self):
        self.first_seen: dict[int, int] = {}
        self.alias: dict[int, int] = {}  # phantom track_id -> the track it is part of
        self.suppressed_detections = 0
        self.suppressed_tracks: dict[int, int] = {}
        self.suppressed_containers = 0

    def update(self, detections: list, frame_id: int) -> list:
        """Returns detections with split boxes collapsed and phantom track_ids
        rewritten to the track they belong to. Untracked boxes (track_id < 0) pass
        through untouched -- with no track there is no age to compare, and nothing
        downstream keys an identity off them."""
        for *_, track_id in detections:
            if track_id >= 0:
                self.first_seen.setdefault(track_id, frame_id)

        present = {tid: box for *box, tid in detections if tid >= 0}
        splits = {
            frozenset((a, b))
            for a in present
            for b in present
            if a < b and self._is_split(present[a], present[b])
        }
        for pair in splits:
            a, b = sorted(pair, key=lambda t: (self.first_seen[t], -_area(present[t])))
            self.alias[b] = a
        # A pair that is both present and no longer a split has genuinely come
        # apart -- two real detections after all, so the alias must not persist.
        for phantom, canonical in list(self.alias.items()):
            if (
                phantom in present
                and canonical in present
                and frozenset((phantom, canonical)) not in splits
            ):
                del self.alias[phantom]

        kept, seen = [], set()
        for x1, y1, x2, y2, track_id in detections:
            if track_id < 0:
                kept.append((x1, y1, x2, y2, track_id))
                continue
            canonical = self._resolve(track_id)
            if canonical in seen:
                self.suppressed_detections += 1
                self.suppressed_tracks[track_id] = self.suppressed_tracks.get(track_id, 0) + 1
                continue
            seen.add(canonical)
            kept.append((x1, y1, x2, y2, canonical))
        containers = [box for box in kept if self._holds_two(box, kept)]
        self.suppressed_containers += len(containers)
        return [box for box in kept if box not in containers]

    def _holds_two(self, box, boxes) -> bool:
        """True when `box` holds at least two other, clearly shorter boxes: a
        union of two bodies, not a third body. A single held box is a player
        standing behind another and is left alone."""
        x1, y1, x2, y2, _ = box
        height = max(1, y2 - y1)
        held = 0
        for ox1, oy1, ox2, oy2, _ in boxes:
            if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
                continue
            if (oy2 - oy1) >= self.HELD_MAX_HEIGHT * height:
                continue
            inside = _intersection(box, (ox1, oy1, ox2, oy2)) / max(1, _area((ox1, oy1, ox2, oy2)))
            held += inside >= self.HELD_INSIDE
        return held >= 2

    def _resolve(self, track_id: int) -> int:
        seen = set()
        while track_id in self.alias and track_id not in seen:
            seen.add(track_id)
            track_id = self.alias[track_id]
        return track_id

    def _is_split(self, box_a, box_b) -> bool:
        overlap = _intersection(box_a, box_b)
        if not overlap:
            return False
        smaller, larger = sorted((box_a, box_b), key=_area)
        if overlap / max(1, _area(smaller)) < self.CONTAINMENT:
            return False
        height = max(1, smaller[3] - smaller[1])
        return (
            abs(smaller[1] - larger[1]) / height <= self.EDGE_TOLERANCE
            and abs(smaller[3] - larger[3]) / height <= self.EDGE_TOLERANCE
        )

    def summary(self) -> str:
        return (
            f"Split detections collapsed (one body, two boxes): "
            f"{self.suppressed_detections} across {len(self.suppressed_tracks)} track(s); "
            f"container boxes dropped (two bodies, three boxes): {self.suppressed_containers}"
        )


class TeamClassifier:
    """Assigns each tracked player to a team based on jersey color.

    Cluster centers are fit once from the first batch of samples, then frozen —
    every later frame assigns players to the nearest *fixed* center rather than
    re-clustering, so a team's identity never flips between frames. A rolling
    average per track ID smooths out noisy single-frame color reads.
    """

    def __init__(
        self,
        k=NUM_TEAM_CLUSTERS,
        fit_after_distinct_tracks=TEAM_FIT_AFTER_SAMPLES,
        min_observations_before_fit_eligible=3,
        smoothing=0.7,
    ):
        self.k = k
        self.fit_after_distinct_tracks = fit_after_distinct_tracks
        self.min_observations_before_fit_eligible = min_observations_before_fit_eligible
        self.smoothing = smoothing
        self.centers = None
        self.cluster_team = []  # cluster index -> team id, set when fit
        self.track_colors = {}
        self.observation_counts = {}

    def observe(self, track_id: int, color):
        if color is None:
            return
        smoothed = self.track_colors.get(track_id)
        smoothed = (
            color if smoothed is None else self.smoothing * smoothed + (1 - self.smoothing) * color
        )
        self.track_colors[track_id] = smoothed
        self.observation_counts[track_id] = self.observation_counts.get(track_id, 0) + 1

        if self.centers is None:
            self._try_fit()

    def team_for(self, track_id: int):
        return self.team_for_color(self.track_colors.get(track_id))

    def team_for_color(self, color):
        """Team of the nearest fixed cluster center for an already-sampled
        color: 0 and 1 are the two most populated colour groups (the kits),
        OTHER_TEAM is everyone else (keepers, officials, staff)."""
        if self.centers is None or color is None:
            return None
        distances = np.linalg.norm(self.centers - color, axis=1)
        return int(self.cluster_team[int(np.argmin(distances))])

    def team_name(self, team_id: int | None) -> str:
        """ "white", "yellow", ... from the team's fitted jersey colour; the
        two teams get distinct names even when both kits read the same."""
        if team_id is None:
            return "unknown"
        if team_id == OTHER_TEAM or self.centers is None:
            return OTHER_TEAM_NAME
        names = [color_name(self.team_color(team)) for team in (0, 1)]
        if names[0] == names[1]:
            names[team_id] = f"{names[team_id]} ({team_id + 1})"
        return names[team_id]

    def team_color(self, team_id: int):
        """The team's actual average jersey color (BGR), for marker rendering."""
        for cluster, team in enumerate(self.cluster_team):
            if team == team_id:
                b, g, r = self.centers[cluster]
                return (int(b), int(g), int(r))
        return None

    def _try_fit(self):
        """Fit once we've seen enough *distinct* tracks with a stable color estimate
        each (not just enough total observations, which a handful of repeatedly-seen
        tracks could satisfy on their own with zero diversity).
        """
        eligible_tracks = [
            track_id
            for track_id, count in self.observation_counts.items()
            if count >= self.min_observations_before_fit_eligible
        ]
        if len(eligible_tracks) < self.fit_after_distinct_tracks:
            return

        data = np.array([self.track_colors[t] for t in eligible_tracks], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        k = min(self.k, len(data))
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k).astype(float)
        centers, counts = self._merge_same_kit(centers, counts)
        # The two most populated groups are the teams; ties in count go to the
        # lower index so the assignment is deterministic for a given fit.
        order = np.argsort(-counts, kind="stable")
        cluster_team = [OTHER_TEAM] * len(centers)
        for team, cluster in enumerate(order[:2]):
            cluster_team[int(cluster)] = team
        self.centers = centers
        self.cluster_team = cluster_team

    def _merge_same_kit(self, centers, counts):
        """One kit under sun and shade can come out as two clusters ~20-30 BGR
        apart; distinct kits are 60+ apart. Merge (count-weighted) until every
        pair is at least MIN_KIT_SEPARATION apart."""
        centers, counts = [c.astype(float) for c in centers], list(counts)
        while len(centers) > 1:
            pairs = [
                (float(np.linalg.norm(centers[i] - centers[j])), i, j)
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            ]
            distance, i, j = min(pairs)
            if distance >= MIN_KIT_SEPARATION:
                break
            total = counts[i] + counts[j]
            centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / total
            counts[i] = total
            del centers[j], counts[j]
        return np.array(centers, dtype=np.float32), np.array(counts)


class JerseySampler:
    """Raw jersey color of a box, sampled straight from this cycle's frame.

    PlayerIdentityManager anchors each player's appearance to this, so it can
    notice when a *live* track's box has quietly moved onto a different body.
    Asked once per (detection, candidate) pair while matching, but the answer
    depends only on the box -- each one is sampled once per cycle.
    """

    def __init__(self, frame):
        self.frame = frame
        self._colors = {}

    def color_of(self, bbox):
        if bbox not in self._colors:
            x1, y1, x2, y2 = bbox
            self._colors[bbox] = extract_jersey_color(self.frame, x1, y1, x2, y2)
        return self._colors[bbox]


class StateManager:
    """Confirms a track as visible only after it's been detected for CONFIRM_CYCLES
    consecutive worker cycles, so a single spurious/noise detection doesn't blink
    onto screen for one frame and vanish. Operates on worker cycles (AI results),
    not display frames, since a single result is redisplayed across several display
    frames via extrapolation.

    Once a track is confirmed, a lone missed detection (motion blur, brief occlusion,
    ordinary detector noise) coasts on its last known box for up to GRACE_CYCLES
    cycles instead of vanishing immediately and having to re-earn CONFIRM_CYCLES from
    scratch — measured on real footage, an unconditional instant-drop caused a
    confirmed track dropout roughly once per worker cycle, i.e. constant flicker.
    """

    CONFIRM_CYCLES = 2
    GRACE_CYCLES = 2

    def __init__(self):
        self.consecutive_counts = {}  # track_id -> consecutive cycles detected
        self.miss_counts = {}  # track_id -> consecutive cycles missed since last seen
        self.last_box = {}  # track_id -> last known (x1, y1, x2, y2), while confirmed

    def update(self, boxes):
        """Call once per worker cycle with that cycle's raw detected boxes. Returns
        the subset of boxes for tracks that have reached CONFIRM_CYCLES, including
        confirmed tracks currently coasting through a within-grace miss."""
        present_ids = set()
        confirmed = []
        for x1, y1, x2, y2, track_id in boxes:
            if track_id < 0:
                confirmed.append((x1, y1, x2, y2, track_id))
                continue
            present_ids.add(track_id)
            count = self.consecutive_counts.get(track_id, 0) + 1
            self.consecutive_counts[track_id] = count
            self.miss_counts[track_id] = 0
            if count >= self.CONFIRM_CYCLES:
                self.last_box[track_id] = (x1, y1, x2, y2)
                confirmed.append((x1, y1, x2, y2, track_id))

        for track_id in list(self.consecutive_counts):
            if track_id in present_ids:
                continue
            if track_id not in self.last_box:
                del self.consecutive_counts[track_id]  # never confirmed, no grace
                continue
            self.miss_counts[track_id] += 1
            if self.miss_counts[track_id] > self.GRACE_CYCLES:
                del self.consecutive_counts[track_id]
                del self.miss_counts[track_id]
                del self.last_box[track_id]
            else:
                x1, y1, x2, y2 = self.last_box[track_id]
                confirmed.append((x1, y1, x2, y2, track_id))

        return confirmed


@dataclass
class PlayerState:
    """Application-level identity that outlives BoT-SORT's own transient track_id.

    Debugging match_4.mp4 found BoT-SORT reassigning a brand-new track_id to the
    same physical player after a single missed frame -- a small/distant,
    borderline-confidence box (e.g. track 95 -> 159, ~9px apart, one frame gap)
    fails the tracker's own IoU/motion match and gets treated as a new track,
    even though `track_buffer` is nowhere near exhausted. Rather than chase that
    inside BoT-SORT's own matching, PlayerIdentityManager reconciles it one layer
    up: a stable `player_id` that downstream code (team classification, the
    confirm/grace display logic, and eventually speed/distance) keys off, so a
    tracker-side ID swap doesn't reset a player's accumulated state.

    `clean_bbox`/`clean_frame` are the last position that was *trusted* -- not
    contaminated by an occlusion, and wearing this player's own jersey -- and
    `vx`/`vy` the velocity measured from trusted positions only.

    `appearance` is the player's jersey color (BGR, slow EMA of sightings that
    agreed with it). `mismatch_streak` counts consecutive clean sightings that
    looked more like some *other* known player's jersey than this one's -- the
    tell that BoT-SORT's track has moved onto another body. `deviation_streak`
    counts consecutive sightings that merely disagreed (shadow, a crowd
    background bleeding into the crop) and `recent_colors` holds those samples.
    """

    player_id: int
    tracker_id: int
    bbox: tuple[int, int, int, int]
    vx: float
    vy: float
    last_seen_frame: int
    status: str  # "active" | "lost"
    first_frame: int = 0
    occluded: bool = False
    clean_bbox: tuple[int, int, int, int] | None = None
    clean_frame: int | None = None
    appearance: np.ndarray | None = None
    mismatch_streak: int = 0
    deviation_streak: int = 0
    recent_colors: list = field(default_factory=list)

    def predicted_center(self, frame_id: int, horizon: int) -> tuple[float, float]:
        """Constant-velocity prediction from the last trusted sighting, with the
        velocity applied for at most `horizon` frames -- players turn and stop,
        so a long linear extrapolation is worse than none."""
        px1, py1, px2, py2 = self.clean_bbox or self.bbox
        gap = min(horizon, max(0, frame_id - (self.clean_frame or self.last_seen_frame)))
        return (px1 + px2) / 2 + self.vx * gap, (py1 + py2) / 2 + self.vy * gap

    def clean_height(self) -> int:
        px1, py1, px2, py2 = self.clean_bbox or self.bbox
        return max(1, py2 - py1)


class PlayerIdentityManager:
    """Reconciles BoT-SORT's transient track_id into a persistent player_id.

    Fast path: a track_id already mapped to a player is a cheap dict lookup --
    no matching needed as long as BoT-SORT keeps reporting the same ID.

    Slow path: unmapped track_ids (first sighting, or BoT-SORT re-detecting a
    player under a new ID) are scored against recently-lost players by predicted
    position -- last known velocity extrapolated over the frame gap -- normalized
    by the player's own bbox height, since a given pixel error means a lot for a
    tiny distant player and nothing for a large close one -- plus jersey-color
    distance to the player's appearance. A bbox size-ratio sanity gate and a
    jersey gate guard against merging two different, merely nearby, players.

    Three things make that slow path survive players converging, which is when a
    wrong merge is both most likely and most damaging (a swapped identity
    silently moves one player's distance and heatmap onto another's):

    * **Jersey gate.** A box that looks clearly more like some other known
      player's jersey than this one's cannot be this player. Relative on
      purpose: absolute color thresholds fail on noisier footage (match_4:
      shadows and crowd backgrounds bleed into the crop, and a yellow kit is
      half-masked as grass), where a box can drift far from its own player's
      color without resembling anyone else's. Free, since jersey color is
      already sampled for team classification.

    * **Joint assignment, not greedy.** All of a cycle's rows are matched to all
      candidates at once (Hungarian, via scipy), so the result doesn't depend on
      detection order and two detections can't compete for the same player.

    * **Occlusion-aware state.** While a player's box overlaps another's -- or
      has swallowed one that vanished into it and is still lost -- its position
      is partly the *other* player's. Velocity is frozen at its last clean value
      and matching predicts forward from the last clean position, so the players
      that emerge from a merge are resolved by the motion they carried into it
      rather than by the merged box's meaningless drift. Occluded players are
      reported in `occluded_player_ids` so downstream analytics can drop those
      samples instead of trusting a contaminated position.

    All of that only ever ran for *unmapped* tracks. The swaps actually seen on
    match_5.mp4 (docs/PLAN.md Plan 2.8) never produce one: the detector returns a
    single box covering two bodies, BoT-SORT keeps its track_id on that box, and
    when the box shrinks back it is on the *other* body -- the goalkeeper's track
    walked off with a defender, a blue player's track walked off with a white
    one. Nothing went lost, so nothing was reconciled. The defense is
    **appearance-anchored identity**: every player carries the jersey color it
    was seen in, every clean single-body sighting is checked against it, and a
    track whose box has looked like someone else's jersey for MISMATCH_CYCLES in
    a row is put back into the same joint assignment alongside the unmapped
    tracks -- against the players it might really be (recently lost ones, other
    suspect tracks' players), with "keep the current owner" as the fallback. A
    track that has just been minted (YOUNG_CYCLES) is re-examined the same way
    whenever something is in flux, because the abandoned body usually gets a
    fresh track a few cycles *before* the stolen one is caught, and that fresh
    identity should yield to the established one.

    Thresholds measured on match_5.mp4 (BGR euclidean, torso median): same-body
    frame-to-frame color noise p99 ~20-35 for on-pitch players, blue vs white
    kits ~75 apart, goalkeeper vs white ~190. match_4.mp4 is far noisier (p95
    40-90 for many players), which is what forced the gates to be relative.
    """

    MAX_LOST_FRAMES = 30  # ~1.25s at 24fps
    MAX_NORM_DIST = 3.0  # (predicted-position error) / bbox_height
    SIZE_RATIO_RANGE = (0.4, 2.5)  # a box that swallowed a second body can be ~2x tall
    VELOCITY_HORIZON = 10  # frames of constant-velocity extrapolation before it's ignored
    APPEARANCE_ALPHA = 0.9  # EMA weight on the existing appearance
    COLOR_MISMATCH = 40.0  # box vs own appearance beyond this: not the jersey we know
    OTHER_JERSEY_RATIO = 0.5  # ...and someone else's jersey this much nearer: it is theirs
    COLOR_SCALE = 40.0  # BGR distance per unit of assignment cost (capped at one unit)
    MISMATCH_CYCLES = 3  # consecutive clean sightings in another jersey before suspect
    ACCEPT_DEVIATION_AFTER = 30  # unclaimed for this long: it was a lighting change, adopt it
    YOUNG_CYCLES = 20  # a freshly minted identity is provisional for this long
    MAX_COST = 3.0  # position + color cost beyond which an unmapped track is not a match
    MAX_TRANSFER_COST = 2.0  # stricter, for taking a track away from its current owner
    MAX_JUDGED_HEIGHT_RATIO = 1.3  # a box this much taller than the player holds two bodies
    MERGED_PAIR_RATIO = 1.4  # ...or this much wider/taller than either of two bodies it covers
    _NO_MATCH = 1e6  # cost standing in for "gate failed"; never an accepted pairing

    def __init__(self):
        self.players: dict[int, PlayerState] = {}
        self.tracker_to_player: dict[int, int] = {}
        self.occluded_player_ids: set[int] = set()
        self._next_id = 1
        self.switch_log = []  # instrumentation: every track_id reconciliation
        self.swap_log = []  # instrumentation: every live track moved to another player
        # instrumentation: ordered player_ids each BoT-SORT track has belonged to.
        self.track_owners: dict[int, list[int]] = {}
        self.jersey_blocked_merges = 0
        self.occluded_cycles = 0
        self.mismatch_cycles = 0  # clean sightings that looked like another player's jersey
        self.adopted_appearances = 0  # deviations nobody claimed, accepted as lighting
        self._appearance_ids = np.empty(0, dtype=int)  # per-cycle index, see _index_appearances
        self._appearances = np.empty((0, 3))

    @property
    def total_players_minted(self) -> int:
        """Count of persistent player_ids ever created (active, lost, or retired)."""
        return self._next_id - 1

    def update(self, detections: list, frame_id: int, sampler=None) -> list:
        """detections: (x1, y1, x2, y2, track_id) tuples from extract_boxes.
        sampler: this cycle's JerseySampler (or None to match on geometry only).

        Returns the same shape with track_id replaced by a persistent
        player_id. Untracked detections (track_id < 0) pass through as -1.
        """
        present_tracker_ids = {tid for *_, tid in detections if tid >= 0}

        # Mark vanished-this-frame players lost *before* reconciling new/unmapped
        # track_ids, so a same-cycle reacquisition (the 95 -> 159 case) sees the
        # just-vanished player as a candidate immediately, not one frame late.
        just_lost = set()
        for player in self.players.values():
            if player.status == "active" and player.tracker_id not in present_tracker_ids:
                player.status = "lost"
                just_lost.add(player.player_id)

        contamination = self._occlusion_flags(detections, frame_id)
        self._index_appearances()
        # A lost player whose last position a live box still covers is not gone,
        # just merged into that box: keep them pending, and *inside* that box as
        # it moves, so they can be matched where the merge ends rather than
        # where it began, however long it lasts and however far it travels. Not
        # on the cycle they vanish, though: their own last motion is still the
        # best guess of where they are, and a crossing is resolved by it.
        for (x1, y1, x2, y2, _), (_, swallowed) in zip(detections, contamination, strict=True):
            for player_id in swallowed - just_lost:
                player = self.players[player_id]
                player.last_seen_frame = frame_id - 1
                player.bbox = player.clean_bbox = (x1, y1, x2, y2)
                player.clean_frame = frame_id - 1
                player.vx = player.vy = 0.0

        results = [None] * len(detections)
        rows = []  # detections whose owner is undecided this cycle
        young = []  # provisional identities, re-examined only when something is in flux
        for i, (x1, y1, x2, y2, tracker_id) in enumerate(detections):
            if tracker_id < 0:
                results[i] = (x1, y1, x2, y2, -1)
                continue
            player_id = self.tracker_to_player.get(tracker_id)
            if player_id is None:
                rows.append(i)
                continue
            self._observe(
                player_id, tracker_id, (x1, y1, x2, y2), frame_id, contamination[i], sampler
            )
            results[i] = (x1, y1, x2, y2, player_id)
            player = self.players[player_id]
            if self._is_suspect(player):
                rows.append(i)
            elif self._is_young(player, frame_id) and self._is_single_body(
                (x1, y1, x2, y2), player, contamination[i]
            ):
                young.append(i)
        # A young identity only yields when there is an established one it could
        # be: a track caught wearing someone else's jersey, or a player who has
        # just vanished (the usual order is that the abandoned body gets its
        # fresh track a few cycles before the stolen one is caught).
        if young and (just_lost or any(results[i] is not None for i in rows)):
            rows.extend(young)

        for i, player_id in self._reconcile_batch(detections, rows, frame_id, sampler):
            x1, y1, x2, y2, tracker_id = detections[i]
            self._observe(
                player_id, tracker_id, (x1, y1, x2, y2), frame_id, contamination[i], sampler
            )
            results[i] = (x1, y1, x2, y2, player_id)

        self._retire_stale(frame_id)
        self.occluded_player_ids = {
            p.player_id for p in self.players.values() if p.status == "active" and p.occluded
        }
        self.occluded_cycles += len(self.occluded_player_ids)
        return results

    def _occlusion_flags(self, detections: list, frame_id: int) -> list:
        """Per detection: which other players contaminate this box right now,
        as (overlaps a live detection, {player_ids swallowed by it}).

        Two ways that happens. The box still overlaps another live detection --
        two players contesting a ball, each box containing some of the other.
        Or the detector collapsed both players into one box, in which case there
        is no second detection to overlap; the tell is a lost player whose last
        position the box covers. That holds for as long as the player stays
        lost (not just the cycle they vanished): the merged box is someone
        else's for the whole merge, and its motion must not be trusted until
        the swallowed player is found again or expires.
        """
        if not detections:
            return []
        boxes = np.array([d[:4] for d in detections], dtype=float)
        live = pairwise_overlap(boxes, boxes)
        np.fill_diagonal(live, False)
        overlaps_live = live.any(axis=1)

        lost = [
            p
            for p in self.players.values()
            if p.status == "lost" and frame_id - p.last_seen_frame <= self.MAX_LOST_FRAMES
        ]
        swallowed = np.zeros((len(detections), len(lost)), dtype=bool)
        if lost:
            swallowed = pairwise_overlap(boxes, [p.bbox for p in lost])
            # a box cannot have swallowed a body several times its own size
            heights = np.maximum(1, boxes[:, 3] - boxes[:, 1])
            ratio = heights[:, None] / np.array([p.clean_height() for p in lost])[None, :]
            swallowed &= (ratio >= self.SIZE_RATIO_RANGE[0]) & (ratio <= self.SIZE_RATIO_RANGE[1])
        return [
            (bool(overlaps_live[i]), {lost[j].player_id for j in np.flatnonzero(swallowed[i])})
            for i in range(len(detections))
        ]

    @staticmethod
    def _is_occluded(contamination, player_id: int) -> bool:
        """A box sitting on top of the track it is itself continuing is just a
        reacquisition, not an occlusion -- only *another* player's presence in
        the box contaminates it."""
        overlaps_live, swallowed = contamination
        return overlaps_live or bool(swallowed - {player_id})

    def _is_single_body(self, bbox, player: PlayerState, contamination: tuple) -> bool:
        """A box that overlaps another detection, or is big enough to hold a
        swallowed player as well, has no position or color of its own to be
        judged by."""
        overlaps_live, swallowed = contamination
        return not overlaps_live and not self._holds_both(bbox, player, swallowed)

    def _holds_both(self, bbox, owner: PlayerState, swallowed: set) -> bool:
        """True when `bbox` is big enough to be `owner` and a swallowed lost
        player side by side or stacked -- two bodies merged into one detection.
        Its color is then a blend and says nothing about who owns the track.
        A single-body-sized box over a lost player's last position is the
        opposite case: one body, and the color says which one."""
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox1, oy1, ox2, oy2 = owner.clean_bbox or owner.bbox
        for player_id in swallowed - {owner.player_id}:
            other = self.players.get(player_id)
            if other is None:
                continue
            qx1, qy1, qx2, qy2 = other.clean_bbox or other.bbox
            wide = w >= self.MERGED_PAIR_RATIO * max(ox2 - ox1, qx2 - qx1)
            tall = h >= self.MERGED_PAIR_RATIO * max(oy2 - oy1, qy2 - qy1)
            if wide or tall:
                return True
        return False

    def _is_suspect(self, player: PlayerState) -> bool:
        return player.mismatch_streak >= self.MISMATCH_CYCLES

    def _is_young(self, player: PlayerState, frame_id: int) -> bool:
        return frame_id - player.first_frame <= self.YOUNG_CYCLES

    def _reconcile_batch(self, detections: list, rows: list, frame_id: int, sampler) -> list:
        """Decide the owner of every undecided detection in this cycle: an
        unmapped track, a suspect track (box no longer wearing its player's
        jersey), or a provisional young one. Candidates are the recently-lost
        players plus the suspects' own current players, so a body swap between
        two live tracks resolves as one exchange. Solved as one assignment so the
        outcome doesn't depend on detection order and no two rows can claim the
        same player. Returns (index, player_id) for every row whose owner is now
        set or changed; a suspect/young row nothing fits keeps its owner, an
        unmapped row nothing fits mints."""
        if not rows:
            return []

        candidates = [
            p
            for p in self.players.values()
            if p.status == "lost" and 0 < frame_id - p.last_seen_frame <= self.MAX_LOST_FRAMES
        ]
        current = {i: self.tracker_to_player.get(detections[i][4]) for i in rows}
        for i in rows:
            player = self.players.get(current[i])
            if player is not None and self._is_suspect(player):
                candidates.append(player)

        assignments = {}
        if candidates:
            costs = np.array(
                [
                    [
                        self._match_cost(detections[i], candidate, frame_id, sampler)
                        for candidate in candidates
                    ]
                    for i in rows
                ]
            )
            row_idx, col_idx = linear_sum_assignment(costs)
            for r, c in zip(row_idx, col_idx, strict=True):
                cost = float(costs[r][c])
                limit = self.MAX_COST if current[rows[r]] is None else self.MAX_TRANSFER_COST
                if cost > limit:
                    continue
                assignments[rows[r]] = (candidates[c], cost)
            self._drop_unaccounted_transfers(assignments, current)

        claimed = {player.player_id for player, _ in assignments.values()}
        resolved = []
        for i in rows:
            tracker_id = detections[i][4]
            matched = assignments.get(i)
            if matched is None:
                if current[i] is None:
                    resolved.append((i, self._mint(tracker_id)))
                continue  # a mapped row nothing else fits keeps its owner
            player, cost = matched
            if player.player_id == current[i]:
                continue
            self._transfer(tracker_id, current[i], player, frame_id, cost, claimed)
            resolved.append((i, player.player_id))
        return resolved

    def _drop_unaccounted_transfers(self, assignments: dict, current: dict) -> None:
        """An established owner only gives its track up when its own body is
        accounted for: some other row of this same assignment takes that owner.
        A box that swallowed a second body can read as the other jersey for as
        long as they overlap (match_4: a white player hidden behind a yellow
        one), and nothing then says which body the track will follow when they
        part. When the displaced owner's jersey shows up on another box -- the
        stolen track's old body, now detected on its own -- the exchange is
        consistent and safe. A provisional young owner needs no such account;
        yielding is what it is for."""
        while True:
            claimed = {player.player_id for player, _ in assignments.values()}
            unaccounted = [
                i
                for i, (player, _) in assignments.items()
                if current[i] is not None
                and player.player_id != current[i]
                and self._is_suspect(self.players[current[i]])
                and current[i] not in claimed
            ]
            if not unaccounted:
                return
            for i in unaccounted:
                del assignments[i]

    def _transfer(self, tracker_id, from_player_id, player, frame_id, cost, claimed) -> None:
        """Hand `tracker_id` to `player`. Its previous owner (if any) goes lost --
        unless another row of this same assignment claimed it -- or is retired
        outright when it was only ever a provisional young identity, so the
        display never keeps a number that was minted for a body someone else
        already owned."""
        # Only unmap the player's old track if it still points at them: after a
        # swap it may be another player's live track by now.
        if self.tracker_to_player.get(player.tracker_id) == player.player_id:
            self.tracker_to_player.pop(player.tracker_id, None)
        self.tracker_to_player[tracker_id] = player.player_id
        if from_player_id is None:
            self.switch_log.append(
                {
                    "frame": frame_id,
                    "player_id": player.player_id,
                    "old_tracker_id": player.tracker_id,
                    "new_tracker_id": tracker_id,
                    "cost": round(cost, 2),
                }
            )
            return
        previous = self.players.get(from_player_id)
        self.swap_log.append(
            {
                "frame": frame_id,
                "tracker_id": tracker_id,
                "from_player_id": from_player_id,
                "to_player_id": player.player_id,
                "cost": round(cost, 2),
            }
        )
        if previous is None or from_player_id in claimed:
            return
        if self._is_young(previous, frame_id) and not self._is_suspect(previous):
            del self.players[from_player_id]
            return
        # It was last truly seen where it was last wearing its own jersey, not
        # wherever the stolen track has dragged its box since.
        previous.bbox = previous.clean_bbox or previous.bbox
        previous.status = "lost"
        previous.last_seen_frame = frame_id

    def _match_cost(self, detection, player: PlayerState, frame_id: int, sampler) -> float:
        """Normalized predicted-position error plus jersey-color distance, or
        _NO_MATCH if any gate rejects the pairing outright."""
        x1, y1, x2, y2, tracker_id = detection
        p_height = player.clean_height()

        size_ratio = max(1, y2 - y1) / p_height
        if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
            return self._NO_MATCH

        color_cost = 0.0
        color = self._row_color((x1, y1, x2, y2), tracker_id, sampler)
        if color is not None and player.appearance is not None:
            color_dist = float(np.linalg.norm(color - player.appearance))
            if self._wears_another_jersey(color, color_dist, player.player_id):
                self.jersey_blocked_merges += 1
                return self._NO_MATCH
            # Bounded: past one unit the jersey gate has already had its say, and a
            # merely odd-colored box must still be able to match on position.
            color_cost = min(1.0, color_dist / self.COLOR_SCALE)

        if frame_id - player.last_seen_frame > self.MAX_LOST_FRAMES:
            return self._NO_MATCH
        pred_cx, pred_cy = player.predicted_center(frame_id, self.VELOCITY_HORIZON)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        norm_dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5 / p_height
        if norm_dist > self.MAX_NORM_DIST:
            return self._NO_MATCH
        return norm_dist + color_cost

    def _index_appearances(self) -> None:
        """Once per cycle: every known appearance as one matrix, so judging a box
        against all players is a single vectorized op rather than a Python loop
        (measured: the loop cost ~1ms per cycle at 25 boxes x 50 players, which
        on a 50fps clip is the difference between dropping frames or not)."""
        known = [(pid, p.appearance) for pid, p in self.players.items() if p.appearance is not None]
        self._appearance_ids = np.array([pid for pid, _ in known], dtype=int)
        self._appearances = (
            np.array([a for _, a in known], dtype=float) if known else np.empty((0, 3))
        )

    def _wears_another_jersey(self, color, own_dist: float, player_id: int) -> bool:
        """True when `color` is not the jersey we know for `player_id` *and*
        clearly resembles some other known player's instead. Noise alone (a
        shadow, a crowd background) moves a color away from its owner without
        moving it onto anyone else."""
        if own_dist <= self.COLOR_MISMATCH or len(self._appearances) == 0:
            return False
        distances = np.linalg.norm(self._appearances - color, axis=1)
        distances[self._appearance_ids == player_id] = np.inf
        return bool(distances.min() < own_dist * self.OTHER_JERSEY_RATIO)

    def _row_color(self, bbox, tracker_id, sampler):
        """The color to judge a row by: a suspect track has several disagreeing
        samples, and their median is steadier than any single frame."""
        player = self.players.get(self.tracker_to_player.get(tracker_id))
        if player is not None and player.recent_colors:
            return np.median(np.array(player.recent_colors), axis=0)
        if sampler is None:
            return None
        return sampler.color_of(bbox)

    def _mint(self, tracker_id: int) -> int:
        new_id = self._next_id
        self._next_id += 1
        self.tracker_to_player[tracker_id] = new_id
        return new_id

    def _observe(
        self,
        player_id: int,
        tracker_id: int,
        bbox: tuple,
        frame_id: int,
        contamination: tuple,
        sampler=None,
    ) -> None:
        existing = self.players.get(player_id)
        overlaps_live, _ = contamination
        occluded = self._is_occluded(contamination, player_id)
        # Does this box still wear the player's jersey? A box overlapping another
        # live detection is nobody's color and is not judged, and neither is one
        # much taller than the body we know: the detector has merged two bodies
        # into it and its color is a blend of both. A single-body box that
        # merely sits where a lost player was *is* judged: if it now wears that
        # jersey and not its own, that is the swap signature, not noise.
        judged = not overlaps_live and sampler is not None
        if judged and existing is not None:
            judged = (bbox[3] - bbox[1]) <= self.MAX_JUDGED_HEIGHT_RATIO * existing.clean_height()
            judged = judged and self._is_single_body(bbox, existing, contamination)
        color = sampler.color_of(bbox) if judged else None

        appearance = existing.appearance if existing is not None else None
        mismatch_streak, deviation_streak, recent_colors = 0, 0, []
        if existing is not None:
            mismatch_streak, deviation_streak = existing.mismatch_streak, existing.deviation_streak
            recent_colors = existing.recent_colors
        consistent = color is not None and appearance is not None
        if consistent:
            own_dist = float(np.linalg.norm(color - appearance))
            consistent = own_dist <= self.COLOR_MISMATCH
            if consistent:
                mismatch_streak, deviation_streak, recent_colors = 0, 0, []
            else:
                deviation_streak += 1
                recent_colors = (recent_colors + [color])[-self.MISMATCH_CYCLES :]
                if self._wears_another_jersey(color, own_dist, player_id):
                    mismatch_streak += 1
                    self.mismatch_cycles += 1
                else:
                    mismatch_streak = 0
        if deviation_streak > self.ACCEPT_DEVIATION_AFTER:
            # Nobody claimed this body in all that time: the jersey did not
            # change, the light on it did. Adopt the new look and move on.
            appearance = np.median(np.array(recent_colors), axis=0)
            mismatch_streak, deviation_streak, recent_colors = 0, 0, []
            consistent = True
            self.adopted_appearances += 1
        # Position is trusted unless the box is contaminated or is plainly on
        # someone else; a merely odd-colored box is still this body, in odd light.
        trusted = not occluded and mismatch_streak == 0

        vx, vy = 0.0, 0.0
        clean_bbox, clean_frame = (bbox, frame_id) if trusted else (None, None)
        first_frame = frame_id
        if existing is not None:
            first_frame = existing.first_frame
            if not trusted:
                vx, vy = existing.vx, existing.vy
                clean_bbox, clean_frame = existing.clean_bbox, existing.clean_frame
            elif existing.clean_frame is not None:
                gap = max(1, frame_id - existing.clean_frame)
                pcx = (existing.clean_bbox[0] + existing.clean_bbox[2]) / 2
                pcy = (existing.clean_bbox[1] + existing.clean_bbox[3]) / 2
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                vx, vy = (cx - pcx) / gap, (cy - pcy) / gap
        if color is not None and not occluded and (appearance is None or consistent):
            appearance = (
                color
                if appearance is None
                else self.APPEARANCE_ALPHA * appearance + (1 - self.APPEARANCE_ALPHA) * color
            )

        owners = self.track_owners.setdefault(tracker_id, [])
        if not owners or owners[-1] != player_id:
            owners.append(player_id)

        self.players[player_id] = PlayerState(
            player_id=player_id,
            tracker_id=tracker_id,
            bbox=bbox,
            vx=vx,
            vy=vy,
            last_seen_frame=frame_id,
            status="active",
            first_frame=first_frame,
            occluded=occluded,
            clean_bbox=clean_bbox or bbox,
            clean_frame=clean_frame if clean_frame is not None else frame_id,
            appearance=appearance,
            mismatch_streak=mismatch_streak,
            deviation_streak=deviation_streak,
            recent_colors=recent_colors,
        )

    def _retire_stale(self, frame_id: int) -> None:
        stale = [
            pid
            for pid, p in self.players.items()
            if p.status == "lost" and frame_id - p.last_seen_frame > self.MAX_LOST_FRAMES
        ]
        for pid in stale:
            del self.players[pid]

    @property
    def contested_tracks(self) -> dict[int, list[int]]:
        """Tracks that changed hands between player_ids. Every such change is a
        deliberate correction (see swap_log); BoT-SORT's own silent swaps leave
        the track's owner unchanged, which is exactly the problem."""
        return {t: owners for t, owners in self.track_owners.items() if len(owners) > 1}

    def summary(self) -> str:
        lines = [
            f"Persistent player IDs minted: {self.total_players_minted}",
            f"track_id -> player_id reconciliations (switches absorbed): {len(self.switch_log)}",
            f"Pairings refused because the box wears another player's jersey: "
            f"{self.jersey_blocked_merges}",
            f"Player-cycles spent occluded (position unreliable): {self.occluded_cycles}",
            f"Clean sightings wearing another player's jersey: {self.mismatch_cycles} "
            f"(unclaimed deviations adopted as lighting: {self.adopted_appearances})",
            f"Live tracks moved to another player (body swaps corrected): {len(self.swap_log)}",
        ]
        for event in self.swap_log[:20]:
            lines.append(
                f"  frame {event['frame']}: track {event['tracker_id']} player "
                f"{event['from_player_id']} -> {event['to_player_id']} (cost={event['cost']})"
            )
        for event in self.switch_log[:20]:
            lines.append(
                f"  frame {event['frame']}: track {event['old_tracker_id']} -> "
                f"{event['new_tracker_id']} kept as player {event['player_id']} "
                f"(cost={event['cost']})"
            )
        if len(self.switch_log) > 20:
            lines.append(f"  ... and {len(self.switch_log) - 20} more")
        return "\n".join(lines)
