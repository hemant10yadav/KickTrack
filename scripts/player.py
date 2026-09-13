from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

NUM_TEAM_CLUSTERS = 3  # 2 teams + referee/other
TEAM_FIT_AFTER_SAMPLES = 25  # jersey-color samples collected before clusters are fixed


def _iou(a: tuple, b: tuple) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    non_green = crop.reshape(-1, 3)[cv2.bitwise_not(green_mask).reshape(-1) > 0]
    pixels = non_green if len(non_green) >= 10 else crop.reshape(-1, 3)
    return np.median(pixels, axis=0)


def boxes_overlap(box_a, box_b, overlap_ratio_threshold=0.2, iou_threshold=0.15) -> bool:
    """True if two boxes overlap enough that jersey-color sampling would risk
    picking up the other player — e.g. two opposing players contesting a ball.
    Checked both as a fraction of the smaller box's area (catches a small box mostly
    swallowed by a bigger one) and as IoU (catches two similarly-sized boxes
    overlapping less than fully).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    if intersection == 0:
        return False

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    overlap_ratio = intersection / min(area_a, area_b)
    iou = intersection / (area_a + area_b - intersection)
    return overlap_ratio > overlap_ratio_threshold or iou > iou_threshold


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
        if self.centers is None:
            return None
        color = self.track_colors.get(track_id)
        if color is None:
            return None
        distances = np.linalg.norm(self.centers - color, axis=1)
        return int(np.argmin(distances))

    def team_color(self, team_id: int):
        """The team's actual average jersey color (BGR), for marker rendering."""
        b, g, r = self.centers[team_id]
        return (int(b), int(g), int(r))

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
        _, _, centers = cv2.kmeans(data, self.k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self.centers = centers


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
    same physical player after a single missed frame — a small/distant,
    borderline-confidence box (e.g. track 95 -> 159, ~9px apart, one frame gap)
    fails the tracker's own IoU/motion match and gets treated as a new track,
    even though `track_buffer` is nowhere near exhausted. Rather than chase that
    inside BoT-SORT's own matching, PlayerIdentityManager reconciles it one layer
    up: a stable `player_id` that downstream code (team classification, the
    confirm/grace display logic, and eventually speed/distance) keys off, so a
    tracker-side ID swap doesn't reset a player's accumulated state.
    """

    player_id: int
    tracker_id: int
    bbox: tuple[int, int, int, int]
    vx: float
    vy: float
    last_seen_frame: int
    status: str  # "active" | "lost"
    team: int | None = None  # established TeamClassifier team_for() reading
    team_mismatch_streak: int = 0  # consecutive cycles team_for() disagreed with `team`


class PlayerIdentityManager:
    """Reconciles BoT-SORT's transient track_id into a persistent player_id.

    Fast path: a track_id already mapped to a player is a cheap dict lookup —
    no matching needed as long as BoT-SORT keeps reporting the same ID.

    Slow path: an unmapped track_id (first sighting, or BoT-SORT re-detecting the
    same player under a new ID) is scored against recently-lost players by
    predicted position — last known velocity extrapolated over the frame gap —
    normalized by the player's own bbox height, since a given pixel error means
    a lot for a tiny distant player and nothing for a large close one. A bbox
    size-ratio sanity gate guards against merging two different, merely nearby,
    players. Within MAX_LOST_FRAMES and MAX_NORM_DIST, it's treated as the same
    player continuing under a new track_id; otherwise a new player is minted.

    The values below are a starting point (deliberately conservative — start
    strict on merges, loosen only against measured false-splits), not tuned
    against real match footage yet.

    Two failure modes this position/size-only logic cannot catch on its own,
    both requiring a `classifier` (a TeamClassifier, keyed by player_id since
    jersey-color sampling runs downstream of this reconciliation):

    - A player gone longer than MAX_LOST_FRAMES: predicted position from
      extrapolated velocity gets unreliable over a long gap, so instead of
      widening the position window, a freshly-minted track_id is held as a
      color-match candidate and retried each cycle against players lost up to
      LONG_LOST_MAX_FRAMES ago, using jersey-color distance instead of position.
    - A crossing/ID swap: two *already-active* track_ids can get their
      detections swapped by BoT-SORT's own matching mid-crossing. Since both
      stay "active" the whole time, the reconciliation above never runs for
      them at all. `_check_swaps` instead watches each active player's
      TeamClassifier team assignment for a sustained mismatch against its own
      established team, and swaps two mutually-mismatched, nearby players'
      track_id mappings back.
    """

    MAX_LOST_FRAMES = 30  # ~1.25s at 24fps
    # Tightened from 3.0 -- at 3.0, a wrong merge visibly renumbered two real
    # on-screen players (ID 45 -> ID 3, ID 47 -> ID 48) on match_5.mp4, and
    # every one of that cluster's bad merges scored between 1.87 and 2.86: i.e.
    # right up against the old ceiling, while every legitimate merge scored
    # well under 1.0. 1.5 was verified to eliminate that entire cluster (no
    # merge in it any longer clears the gate) while staying well inside the
    # existing player-ID-count regression ceilings on all three fixture clips
    # (tests/test_tracking_quality.py -- minted 33/37/34 vs ceilings of
    # 70/45/65, i.e. comfortable headroom, not a knife's-edge trade). Loosen
    # only against a *measured* false-split rate on real footage, never back
    # toward 3.0 without the same visual check that caught this.
    MAX_NORM_DIST = 1.5  # (predicted-position error) / bbox_height
    SIZE_RATIO_RANGE = (0.5, 2.0)

    # A crowded/occluded moment can put several real, distinct players within
    # MAX_NORM_DIST of the same predicted position at *the same instant*.
    # Matching one unmapped track_id at a time to whichever lost player looks
    # nearest is a greedy, order-dependent choice that can steal the right
    # candidate out from under a different track_id processed later in the
    # same frame. AMBIGUITY_MARGIN requires the best candidate to be clearly
    # better than the runner-up (>=30% closer) before trusting the match at
    # all -- when two simultaneous candidates are close enough to call, this
    # refuses to guess and mints a new player_id instead, since a wrong silent
    # merge corrupts two real players' analytics while a wrongly-split
    # identity is at least visible and recoverable. Note this only catches
    # *simultaneous* ambiguity within one frame's candidate pool -- it does
    # NOT catch a single, sequential misattribution to one lost player at a
    # time with no competing candidate that instant (that failure mode is
    # what MAX_NORM_DIST above was tightened for, after being caught visually
    # on match_5.mp4: ID 45 -> ID 3, ID 47 -> ID 48).
    AMBIGUITY_MARGIN = 1.3

    _NO_MATCH_COST = 1e6  # sentinel cost for a disallowed (gate-failed) pairing

    LONG_LOST_MAX_FRAMES = 90  # ~3.75s at 24fps -- color-gated, not position-gated
    COLOR_MAX_DIST = 40.0  # max BGR L2 distance between smoothed jersey colors
    # Physical plausibility for the long-gap color path, in bbox-heights per
    # worker cycle: a flat-out sprint is ~9 m/s ~= 5 body-heights/s, i.e. 0.1
    # h/frame at 50fps or 0.2 at 25fps. 0.2 is the conservative (25fps) bound
    # so it never rejects a real sprint on slower footage; it still confines a
    # 90-frame reappearance to a region rather than the whole pitch.
    MAX_TRAVEL_PER_FRAME = 0.2
    # Constant-velocity extrapolation is only trustworthy for a moment: players
    # cut, stop and turn, so over a 20-30 frame gap `v * gap` can land far from
    # where a player who simply slowed down actually reappears (seen on
    # match_5.mp4: a reappearance 0.64 h from its last position, 28 frames
    # later, went unmatched because the extrapolated point had run away from
    # it). Extrapolate at most this many frames of motion, then hold.
    VELOCITY_HORIZON_FRAMES = 10

    # Duplicate-track aliasing. BoT-SORT sometimes runs TWO tracks on one
    # physical player and alternates which one it emits each frame; both
    # track_ids get their own player_id here, so the on-screen number flips
    # back and forth (match_5.mp4: ID 45 <-> ID 3 eight times in 90 frames)
    # or two labels are drawn on one body (ID 3 + ID 22, ID 58 + ID 1). The
    # reconciliation paths above only ever see *unmapped* track_ids, so two
    # already-mapped players on one body are never merged by them. Instead:
    # two players whose boxes overlap this heavily, at similar size and not
    # on different teams, on ALIAS_CONFIRM_FRAMES frames within ALIAS_WINDOW
    # are one person -- UNLESS they were seen clearly apart (both observed,
    # IoU < ALIAS_APART_IOU) inside that window: two real players lining up
    # along the camera axis are seen apart before and after the overlap; a
    # duplicate-track pair never is. That veto is what keeps this from fusing
    # two real teammates during a shoulder-to-shoulder run.
    ALIAS_IOU = 0.55
    ALIAS_APART_IOU = 0.3
    ALIAS_CONFIRM_FRAMES = 4
    ALIAS_WINDOW = 45

    SWAP_MISMATCH_CYCLES = 3  # consecutive cycles of team disagreement before acting
    SWAP_PROXIMITY_RATIO = 3.0  # max center distance / bbox height to call it a crossing
    SWAP_COOLDOWN_FRAMES = 30  # ~0.6s at 50fps -- see _swap() for why this alone isn't the fix

    def __init__(self, classifier=None):
        self.players: dict[int, PlayerState] = {}
        self.tracker_to_player: dict[int, int] = {}
        self._next_id = 1
        self.switch_log = []  # instrumentation: every track_id reconciliation
        self.swap_log = []  # instrumentation: every crossing swap corrected
        self.classifier = classifier
        self._color_pending: dict[int, int] = {}  # player_id -> frame first minted
        self._last_swap_frame: dict[int, int] = {}  # player_id -> frame of its last swap
        self.ambiguous_log = []  # instrumentation: reconciliations refused as too close to call
        self.alias_log = []  # instrumentation: duplicate-track pairs merged into one player
        self._alias_evidence: dict[tuple[int, int], list[int]] = {}  # pair -> overlap frames
        self._alias_apart: dict[tuple[int, int], int] = {}  # pair -> last frame seen apart

    @property
    def total_players_minted(self) -> int:
        """Count of persistent player_ids ever created (active, lost, or retired)."""
        return self._next_id - 1

    def update(self, detections: list, frame_id: int) -> list:
        """detections: (x1, y1, x2, y2, track_id) tuples from extract_boxes.

        Returns the same shape with track_id replaced by a persistent
        player_id. Untracked detections (track_id < 0) pass through as -1.
        """
        present_tracker_ids = {tid for *_, tid in detections if tid >= 0}

        # Mark vanished-this-frame players lost *before* reconciling new/unmapped
        # track_ids, so a same-cycle reacquisition (the 95 -> 159 case) sees the
        # just-vanished player as a candidate immediately, not one frame late.
        for player in self.players.values():
            if player.status == "active" and player.tracker_id not in present_tracker_ids:
                player.status = "lost"

        self._retry_color_pending(frame_id)

        unmapped = []  # (tracker_id, bbox) needing reconciliation, this frame
        pass_through = []  # (x1, y1, x2, y2, tracker_id) already mapped or untracked
        for x1, y1, x2, y2, tracker_id in detections:
            if tracker_id < 0 or self.tracker_to_player.get(tracker_id) is not None:
                pass_through.append((x1, y1, x2, y2, tracker_id))
                continue
            unmapped.append((tracker_id, (x1, y1, x2, y2)))

        assignments = self._reconcile_batch(unmapped, frame_id, present_tracker_ids)

        results = []
        for x1, y1, x2, y2, tracker_id in pass_through:
            if tracker_id < 0:
                results.append((x1, y1, x2, y2, -1))
                continue
            player_id = self.tracker_to_player[tracker_id]
            self._observe(player_id, tracker_id, (x1, y1, x2, y2), frame_id)
            results.append((x1, y1, x2, y2, player_id))
        for tracker_id, bbox in unmapped:
            player_id = assignments[tracker_id]
            self._observe(player_id, tracker_id, bbox, frame_id)
            results.append((*bbox, player_id))

        results = self._check_aliases(results, frame_id)

        if self.classifier is not None:
            self._check_swaps(frame_id)

        self._retire_stale(frame_id)
        return results

    def _check_aliases(self, results: list, frame_id: int) -> list:
        """Merge duplicate-track pairs (see ALIAS_* above) and rewrite this
        frame's results so the dropped player_id never reaches the caller."""
        recent = [p for p in self.players.values() if frame_id - p.last_seen_frame <= 2]
        merges = []
        for i, a in enumerate(recent):
            for b in recent[i + 1 :]:
                a_now, b_now = a.last_seen_frame == frame_id, b.last_seen_frame == frame_id
                if not (a_now or b_now):
                    continue
                pair = (min(a.player_id, b.player_id), max(a.player_id, b.player_id))
                overlap = _iou(a.bbox, b.bbox)
                if a_now and b_now and overlap < self.ALIAS_APART_IOU:
                    self._alias_apart[pair] = frame_id
                    self._alias_evidence.pop(pair, None)
                    continue
                if overlap < self.ALIAS_IOU:
                    continue
                a_h = max(1, a.bbox[3] - a.bbox[1])
                b_h = max(1, b.bbox[3] - b.bbox[1])
                if not (self.SIZE_RATIO_RANGE[0] <= a_h / b_h <= self.SIZE_RATIO_RANGE[1]):
                    continue
                if self.classifier is not None:
                    team_a = self.classifier.team_for(a.player_id)
                    team_b = self.classifier.team_for(b.player_id)
                    if team_a is not None and team_b is not None and team_a != team_b:
                        continue
                frames = self._alias_evidence.setdefault(pair, [])
                frames.append(frame_id)
                frames[:] = [f for f in frames if frame_id - f <= self.ALIAS_WINDOW]
                apart = self._alias_apart.get(pair)
                if apart is not None and frame_id - apart <= self.ALIAS_WINDOW:
                    continue
                if len(frames) >= self.ALIAS_CONFIRM_FRAMES:
                    merges.append(pair)

        remap = {}
        for keep, drop in merges:
            if (
                keep in remap
                or drop in remap
                or keep not in self.players
                or drop not in self.players
            ):
                continue
            self._alias(keep, drop, frame_id)
            remap[drop] = keep
        if not remap:
            return results

        rewritten, seen = [], set()
        for x1, y1, x2, y2, pid in results:
            pid = remap.get(pid, pid)
            if pid >= 0 and pid in seen:
                pid = -1  # the duplicate's box: same person, already reported this frame
            elif pid >= 0:
                seen.add(pid)
            rewritten.append((x1, y1, x2, y2, pid))
        return rewritten

    def _alias(self, keep: int, drop: int, frame_id: int) -> None:
        keep_p, drop_p = self.players[keep], self.players.pop(drop)
        for tracker_id, pid in list(self.tracker_to_player.items()):
            if pid == drop:
                self.tracker_to_player[tracker_id] = keep
        if drop_p.last_seen_frame >= keep_p.last_seen_frame:
            keep_p.tracker_id = drop_p.tracker_id
            keep_p.bbox = drop_p.bbox
            keep_p.vx, keep_p.vy = drop_p.vx, drop_p.vy
            keep_p.last_seen_frame = drop_p.last_seen_frame
            keep_p.status = drop_p.status
        self._color_pending.pop(drop, None)
        self._last_swap_frame.pop(drop, None)
        if self.classifier is not None:
            self.classifier.track_colors.pop(drop, None)
            self.classifier.observation_counts.pop(drop, None)
        for pair in [p for p in self._alias_evidence if drop in p]:
            del self._alias_evidence[pair]
        for pair in [p for p in self._alias_apart if drop in p]:
            del self._alias_apart[pair]
        self.alias_log.append({"frame": frame_id, "kept": keep, "dropped": drop})

    def _reconcile_batch(
        self, unmapped: list[tuple[int, tuple]], frame_id: int, present_tracker_ids: set[int]
    ) -> dict[int, int]:
        """Matches every unmapped track_id seen this frame against every
        recently-lost player at once (a bipartite assignment, solved optimally
        via the Hungarian algorithm), instead of matching one track_id at a
        time against whichever lost player looks nearest.

        The one-at-a-time approach used to let an earlier track_id in the same
        frame greedily claim a marginal candidate before a later, better-suited
        track_id in the *same* frame ever got to consider it -- purely an
        artifact of detection order, not evidence about which pairing was
        actually right. Solving all of this frame's candidates jointly removes
        that ordering artifact. It does NOT remove genuine ambiguity (two real
        players legitimately equidistant from the same predicted spot) --
        AMBIGUITY_MARGIN below refuses to guess in that case instead.

        Returns tracker_id -> player_id for every entry in `unmapped` (minting
        a new player_id for anything that didn't get a confident match).
        """
        candidates = [
            player
            for player in self.players.values()
            if player.status == "lost"
            and 0 < frame_id - player.last_seen_frame <= self.MAX_LOST_FRAMES
            # A "lost" player's own original tracker_id can itself reappear in
            # this same frame (an ordinary pass-through reacquisition -- see
            # update()) without ever going through this method. If it's also
            # left in this candidate list, a completely different unmapped
            # detection can get matched to this same player_id, and then the
            # merge's tracker_to_player rewrite collides with the pass-through
            # reacquisition of the very same player later in the same frame
            # (observed as a KeyError on real footage: two detections in one
            # frame both resolving to the same, about-to-reactivate player).
            and player.tracker_id not in present_tracker_ids
        ]

        assignments: dict[int, int] = {}
        matched_rows: set[int] = set()

        if unmapped and candidates:
            # A finite sentinel, not np.inf: scipy's linear_sum_assignment can raise
            # "cost matrix is infeasible" for certain infinite-cost configurations
            # (e.g. a row with no valid candidate at all). A large finite value still
            # makes Hungarian avoid it whenever any real alternative exists, without
            # that edge case -- _NO_MATCH_COST is filtered back out explicitly below.
            cost = np.full((len(unmapped), len(candidates)), self._NO_MATCH_COST, dtype=float)
            for i, (_, bbox) in enumerate(unmapped):
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                height = max(1, bbox[3] - bbox[1])
                for j, player in enumerate(candidates):
                    gap = frame_id - player.last_seen_frame
                    px1, py1, px2, py2 = player.bbox
                    p_height = max(1, py2 - py1)
                    size_ratio = height / p_height
                    if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
                        continue
                    horizon = min(gap, self.VELOCITY_HORIZON_FRAMES)
                    pred_cx = (px1 + px2) / 2 + player.vx * horizon
                    pred_cy = (py1 + py2) / 2 + player.vy * horizon
                    norm_dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5 / p_height
                    if norm_dist > self.MAX_NORM_DIST:
                        continue
                    cost[i, j] = norm_dist

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind, strict=True):
                best = cost[i, j]
                if best >= self._NO_MATCH_COST:
                    continue  # this detection had no valid candidate at all

                # Ambiguity gate: refuse the match unless it's clearly better than
                # every other candidate this same detection could have gone to.
                other_costs = np.delete(cost[i], j)
                runner_up = other_costs.min() if other_costs.size else np.inf
                if runner_up < self._NO_MATCH_COST and runner_up < best * self.AMBIGUITY_MARGIN:
                    tracker_id, _ = unmapped[i]
                    self.ambiguous_log.append(
                        {
                            "frame": frame_id,
                            "tracker_id": tracker_id,
                            "best_norm_dist": round(float(best), 2),
                            "runner_up_norm_dist": round(float(runner_up), 2),
                        }
                    )
                    continue  # too close to call -- mint a new player_id instead of guessing

                tracker_id, _ = unmapped[i]
                player = candidates[j]
                self.switch_log.append(
                    {
                        "frame": frame_id,
                        "player_id": player.player_id,
                        "old_tracker_id": player.tracker_id,
                        "new_tracker_id": tracker_id,
                        "norm_dist": round(float(best), 2),
                        # confidence: how much better the accepted match was than the
                        # closest alternative this same detection could have gone to --
                        # low values (near AMBIGUITY_MARGIN) mean it barely cleared the
                        # ambiguity gate and downstream analytics may want to discount it.
                        "confidence_margin": (
                            round(float(runner_up / best), 2)
                            if runner_up < self._NO_MATCH_COST and best > 0
                            else None
                        ),
                    }
                )
                # player.tracker_id is stale once a player is "lost" -- BoT-SORT can
                # and does recycle small integer track_ids for a brand-new, unrelated,
                # currently-active physical player in the meantime. Only clear that
                # slot if it's still actually this player's own mapping; otherwise
                # popping it would silently corrupt a different, currently-valid
                # player's tracker_id -> player_id entry (observed as a KeyError
                # later in the same frame's already-mapped detections on real
                # footage once this path started firing more often).
                if self.tracker_to_player.get(player.tracker_id) == player.player_id:
                    self.tracker_to_player.pop(player.tracker_id, None)
                self.tracker_to_player[tracker_id] = player.player_id
                assignments[tracker_id] = player.player_id
                matched_rows.add(i)

        for i, (tracker_id, _) in enumerate(unmapped):
            if i in matched_rows:
                continue
            new_id = self._next_id
            self._next_id += 1
            self.tracker_to_player[tracker_id] = new_id
            if self.classifier is not None:
                self._color_pending[new_id] = frame_id
            assignments[tracker_id] = new_id

        return assignments

    def _observe(self, player_id: int, tracker_id: int, bbox: tuple, frame_id: int) -> None:
        existing = self.players.get(player_id)
        if existing is not None:
            gap = max(1, frame_id - existing.last_seen_frame)
            pcx = (existing.bbox[0] + existing.bbox[2]) / 2
            pcy = (existing.bbox[1] + existing.bbox[3]) / 2
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            vx, vy = (cx - pcx) / gap, (cy - pcy) / gap
            team, team_mismatch_streak = existing.team, existing.team_mismatch_streak
        else:
            vx, vy = 0.0, 0.0
            team, team_mismatch_streak = None, 0
        self.players[player_id] = PlayerState(
            player_id=player_id,
            tracker_id=tracker_id,
            bbox=bbox,
            vx=vx,
            vy=vy,
            last_seen_frame=frame_id,
            status="active",
            team=team,
            team_mismatch_streak=team_mismatch_streak,
        )

    def _retire_stale(self, frame_id: int) -> None:
        # Without a classifier there's no color-matching to wait for, so keep the
        # original short window; with one, hold lost players around longer so a
        # late color-matched candidate (see _retry_color_pending) still has a
        # target to merge into.
        limit = self.LONG_LOST_MAX_FRAMES if self.classifier is not None else self.MAX_LOST_FRAMES
        stale = [
            pid
            for pid, p in self.players.items()
            if p.status == "lost" and frame_id - p.last_seen_frame > limit
        ]
        for pid in stale:
            del self.players[pid]
            self._color_pending.pop(pid, None)
            for pair in [p for p in self._alias_evidence if pid in p]:
                del self._alias_evidence[pair]
            for pair in [p for p in self._alias_apart if pid in p]:
                del self._alias_apart[pair]

    def _retry_color_pending(self, frame_id: int) -> None:
        """Retry color-based matching for track_ids that were minted as brand-new
        players (no position match within MAX_LOST_FRAMES) but might still be a
        player who's been gone longer -- position prediction alone isn't reliable
        over a long gap, so this waits for the candidate to get its first
        jersey-color sample (recorded downstream, keyed by player_id) and then
        compares it against everyone still lost within LONG_LOST_MAX_FRAMES.
        """
        if not self.classifier or not self._color_pending:
            return
        resolved = []
        claimed_targets: set[int] = set()
        for player_id, created_frame in self._color_pending.items():
            candidate = self.players.get(player_id)
            if candidate is None or frame_id - created_frame > self.LONG_LOST_MAX_FRAMES:
                resolved.append(player_id)
                continue
            if candidate.status != "active":
                # A candidate that has itself vanished has nothing current to
                # merge; folding a lost track into a lost target just left the
                # target "lost" with a stale tracker_id, so the *next* pending
                # candidate could merge into it too -- two tracker_ids mapped to
                # one player_id (seen: tracks 43 and 45 both -> player 37 in
                # consecutive frames). Wait for it to come back, or retire.
                continue
            color = self.classifier.track_colors.get(player_id)
            if color is None:
                continue  # no jersey-color sample yet -- keep waiting
            match_id = self._find_color_match(candidate, color, frame_id, claimed_targets)
            if match_id is not None:
                self._merge_into(candidate, match_id, frame_id)
                claimed_targets.add(match_id)
                resolved.append(player_id)
        for player_id in resolved:
            del self._color_pending[player_id]

    def _find_color_match(
        self, candidate: PlayerState, color, frame_id: int, claimed_targets: set[int]
    ):
        """Jersey color is a *gate* here, not the ranking. After tightening
        MAX_NORM_DIST this path carried ~90% of all merges on match_5.mp4, and
        it used to accept the closest *color* anywhere on the pitch with no
        position check at all -- but every teammate's smoothed color sits
        within COLOR_MAX_DIST of every other's, so ranking by color between two
        lost teammates was a coin flip, and a same-kit player could be merged
        into a lost teammate on the far side of the pitch. Now a candidate must
        also be physically reachable from where the lost player was last seen
        (MAX_TRAVEL_PER_FRAME * gap, plus slack), the plausible ones are ranked
        by that travel distance, and a too-close-to-call runner-up refuses the
        merge, same as the short-gap path.

        `claimed_targets`: lost players already merged into this cycle -- two
        pending candidates once both merged into the same player 37 in one
        frame (two tracker_ids -> one player_id -> a duplicate on screen).
        """
        cx = (candidate.bbox[0] + candidate.bbox[2]) / 2
        cy = (candidate.bbox[1] + candidate.bbox[3]) / 2
        c_height = max(1, candidate.bbox[3] - candidate.bbox[1])
        plausible = []  # (travel_norm_dist, player_id)
        for player in self.players.values():
            if player.status != "lost" or player.player_id == candidate.player_id:
                continue
            if player.player_id in claimed_targets:
                continue
            gap = frame_id - player.last_seen_frame
            # Short gaps are already handled by position matching in _reconcile;
            # this path only covers the window position matching gave up on.
            if gap <= self.MAX_LOST_FRAMES or gap > self.LONG_LOST_MAX_FRAMES:
                continue
            old_color = self.classifier.track_colors.get(player.player_id)
            if old_color is None:
                continue
            p_height = max(1, player.bbox[3] - player.bbox[1])
            size_ratio = c_height / p_height
            if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
                continue
            color_dist = float(
                np.linalg.norm(np.asarray(color, dtype=float) - np.asarray(old_color, dtype=float))
            )
            if color_dist > self.COLOR_MAX_DIST:
                continue
            pcx, pcy = (player.bbox[0] + player.bbox[2]) / 2, (player.bbox[1] + player.bbox[3]) / 2
            travel = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / p_height
            if travel > self.MAX_TRAVEL_PER_FRAME * gap + self.MAX_NORM_DIST:
                continue
            plausible.append((travel, player.player_id))

        if not plausible:
            return None
        plausible.sort()
        best_travel, best_id = plausible[0]
        if len(plausible) > 1:
            runner_up = plausible[1][0]
            if runner_up < best_travel * self.AMBIGUITY_MARGIN:
                self.ambiguous_log.append(
                    {
                        "frame": frame_id,
                        "tracker_id": candidate.tracker_id,
                        "best_norm_dist": round(best_travel, 2),
                        "runner_up_norm_dist": round(runner_up, 2),
                        "note": "long-gap color match",
                    }
                )
                return None
        return best_id

    def _merge_into(self, candidate: PlayerState, target_player_id: int, frame_id: int) -> None:
        """Fold a color-pending candidate into the long-lost player it matched,
        so future frames report the old, established player_id instead."""
        target = self.players[target_player_id]
        old_tracker_id = target.tracker_id
        # Same recycled-track_id guard as _reconcile_batch: only clear the
        # target's old slot if it is still the target's own mapping.
        if self.tracker_to_player.get(old_tracker_id) == target_player_id:
            self.tracker_to_player.pop(old_tracker_id, None)
        self.tracker_to_player[candidate.tracker_id] = target_player_id
        target.tracker_id = candidate.tracker_id
        target.bbox = candidate.bbox
        target.vx, target.vy = candidate.vx, candidate.vy
        target.last_seen_frame = candidate.last_seen_frame
        target.status = candidate.status
        self.switch_log.append(
            {
                "frame": frame_id,
                "player_id": target_player_id,
                "old_tracker_id": old_tracker_id,
                "new_tracker_id": candidate.tracker_id,
                "note": "long-gap color match",
            }
        )
        del self.players[candidate.player_id]

    def _check_swaps(self, frame_id: int) -> None:
        """Detect and correct a crossing-players ID swap: two track_ids that stayed
        continuously active the whole time (so _reconcile never saw them as
        unmapped) but got their detections swapped mid-crossing by BoT-SORT's own
        matching. A sustained team-classification mismatch on both sides, plus
        the two players being close together, is treated as evidence of a swap.

        Only corrects the mapping going forward -- frames already emitted during
        the SWAP_MISMATCH_CYCLES it took to detect the swap stay as reported.

        SWAP_COOLDOWN_FRAMES excludes either player from re-evaluation for a
        window right after a swap. This is a backstop, not the primary fix --
        the real fix is _swap() carrying the classifier's smoothed color state
        along with the tracker_id mapping (see its docstring); this cooldown
        just protects against any residual single-cycle noise on top of that.
        """
        mismatched = []
        for player in self.players.values():
            if player.status != "active":
                continue
            if frame_id - self._last_swap_frame.get(player.player_id, -(10**9)) < (
                self.SWAP_COOLDOWN_FRAMES
            ):
                continue
            team = self.classifier.team_for(player.player_id)
            if team is None:
                continue
            if player.team is None:
                player.team = team
                player.team_mismatch_streak = 0
                continue
            if team == player.team:
                player.team_mismatch_streak = 0
                continue
            player.team_mismatch_streak += 1
            if player.team_mismatch_streak >= self.SWAP_MISMATCH_CYCLES:
                mismatched.append((player, team))

        swapped_ids = set()
        for i, (player_a, current_a) in enumerate(mismatched):
            if player_a.player_id in swapped_ids:
                continue
            for player_b, current_b in mismatched[i + 1 :]:
                if player_b.player_id in swapped_ids:
                    continue
                if current_a != player_b.team or current_b != player_a.team:
                    continue  # not a mutual swap between exactly these two teams
                height = max(
                    1, player_a.bbox[3] - player_a.bbox[1], player_b.bbox[3] - player_b.bbox[1]
                )
                cx_a = (player_a.bbox[0] + player_a.bbox[2]) / 2
                cy_a = (player_a.bbox[1] + player_a.bbox[3]) / 2
                cx_b = (player_b.bbox[0] + player_b.bbox[2]) / 2
                cy_b = (player_b.bbox[1] + player_b.bbox[3]) / 2
                dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                if dist / height > self.SWAP_PROXIMITY_RATIO:
                    continue
                self._swap(player_a, player_b, frame_id)
                swapped_ids.add(player_a.player_id)
                swapped_ids.add(player_b.player_id)
                break

    def _swap(self, player_a: PlayerState, player_b: PlayerState, frame_id: int) -> None:
        """Swaps the tracker_id mapping AND the classifier's smoothed color
        history between the two players.

        The color-history swap is not optional bookkeeping -- without it, this
        was observed to oscillate forever on real footage: track_colors[a] had
        already been polluted with several cycles of player b's jersey color
        (that pollution is exactly what raised the mismatch that triggered this
        swap), and TeamClassifier.observe()'s heavy smoothing (0.7) takes many
        more cycles to decay it back out. Left in place after only fixing the
        tracker_id mapping, team_for(a) kept reporting the wrong team for
        several more cycles post-swap, which re-triggered another (this time
        wrong, undoing) swap 3 cycles later -- and so on indefinitely. Moving
        the color history with the identity fixes team_for() immediately
        instead of waiting on it to reconverge.
        """
        self.tracker_to_player[player_a.tracker_id], self.tracker_to_player[player_b.tracker_id] = (
            player_b.player_id,
            player_a.player_id,
        )
        player_a.tracker_id, player_b.tracker_id = player_b.tracker_id, player_a.tracker_id
        player_a.team_mismatch_streak = 0
        player_b.team_mismatch_streak = 0
        self._last_swap_frame[player_a.player_id] = frame_id
        self._last_swap_frame[player_b.player_id] = frame_id

        colors = self.classifier.track_colors
        counts = self.classifier.observation_counts
        a_id, b_id = player_a.player_id, player_b.player_id
        color_a, color_b = colors.get(a_id), colors.get(b_id)
        for target_id, value in ((a_id, color_b), (b_id, color_a)):
            if value is None:
                colors.pop(target_id, None)
            else:
                colors[target_id] = value
        counts[a_id], counts[b_id] = counts.get(b_id, 0), counts.get(a_id, 0)

        self.swap_log.append(
            {"frame": frame_id, "player_a": player_a.player_id, "player_b": player_b.player_id}
        )

    def summary(self) -> str:
        lines = [
            f"Persistent player IDs minted: {self.total_players_minted}",
            f"track_id -> player_id reconciliations (switches absorbed): {len(self.switch_log)}",
            f"crossing-player ID swaps corrected: {len(self.swap_log)}",
            f"ambiguous reconciliations refused (new player_id minted instead): "
            f"{len(self.ambiguous_log)}",
            f"duplicate-track pairs aliased into one player: {len(self.alias_log)}",
        ]
        for event in self.switch_log[:20]:
            detail = event.get("note") or f"norm_dist={event.get('norm_dist')}"
            margin = event.get("confidence_margin")
            if margin is not None:
                detail += f", confidence_margin={margin}"
            lines.append(
                f"  frame {event['frame']}: track {event['old_tracker_id']} -> "
                f"{event['new_tracker_id']} kept as player {event['player_id']} ({detail})"
            )
        if len(self.switch_log) > 20:
            lines.append(f"  ... and {len(self.switch_log) - 20} more")
        for event in self.swap_log[:20]:
            lines.append(
                f"  frame {event['frame']}: swapped tracker mapping between "
                f"player {event['player_a']} and player {event['player_b']}"
            )
        if len(self.swap_log) > 20:
            lines.append(f"  ... and {len(self.swap_log) - 20} more")
        for event in self.ambiguous_log[:20]:
            path = f", {event['note']}" if event.get("note") else ""
            lines.append(
                f"  frame {event['frame']}: track {event['tracker_id']} refused "
                f"(best_norm_dist={event['best_norm_dist']}, "
                f"runner_up={event['runner_up_norm_dist']}{path}) -- minted new player instead"
            )
        if len(self.ambiguous_log) > 20:
            lines.append(f"  ... and {len(self.ambiguous_log) - 20} more")
        for event in self.alias_log[:20]:
            lines.append(
                f"  frame {event['frame']}: player {event['dropped']} was a duplicate track of "
                f"player {event['kept']} -- merged"
            )
        if len(self.alias_log) > 20:
            lines.append(f"  ... and {len(self.alias_log) - 20} more")
        return "\n".join(lines)
