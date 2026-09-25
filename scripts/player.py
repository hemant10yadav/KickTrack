from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

NUM_TEAM_CLUSTERS = 3  # 2 teams + referee/other
TEAM_FIT_AFTER_SAMPLES = 25  # jersey-color samples collected before clusters are fixed


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
        return self.team_for_color(self.track_colors.get(track_id))

    def team_for_color(self, color):
        """Nearest fixed cluster center for an already-sampled color. Separate
        from team_for so a color sampled outside the per-track history (a
        not-yet-identified box being reconciled) can be classified too."""
        if self.centers is None or color is None:
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


class TeamGate:
    """Team identity for one worker cycle, in the two shapes identity
    reconciliation needs: the team already learned for a known player, and the
    team of a box that has no identity yet (sampled straight from the frame).

    Two players in different jerseys can never be the same person, so this is
    the cheapest hard constraint available against merging one player's track
    into another's during a crossing -- the failure that silently moves one
    player's distance and heatmap onto their opponent.
    """

    def __init__(self, classifier: "TeamClassifier", frame):
        self.classifier = classifier
        self.frame = frame
        # of_box is asked once per (detection, candidate) pair while matching, but
        # the answer depends only on the box -- sample each one once per cycle.
        self._sampled = {}

    def of_player(self, player_id: int):
        return self.classifier.team_for(player_id)

    def of_box(self, bbox) -> int | None:
        if bbox not in self._sampled:
            x1, y1, x2, y2 = bbox
            color = extract_jersey_color(self.frame, x1, y1, x2, y2)
            self._sampled[bbox] = self.classifier.team_for_color(color)
        return self._sampled[bbox]


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

    `clean_bbox`/`clean_frame` are the last position that was *not* contaminated
    by an occlusion, and `vx`/`vy` the velocity measured from clean positions
    only -- see PlayerIdentityManager for why the occluded ones can't be used.
    """

    player_id: int
    tracker_id: int
    bbox: tuple[int, int, int, int]
    vx: float
    vy: float
    last_seen_frame: int
    status: str  # "active" | "lost"
    team: int | None = None
    occluded: bool = False
    clean_bbox: tuple[int, int, int, int] | None = None
    clean_frame: int | None = None


class PlayerIdentityManager:
    """Reconciles BoT-SORT's transient track_id into a persistent player_id.

    Fast path: a track_id already mapped to a player is a cheap dict lookup --
    no matching needed as long as BoT-SORT keeps reporting the same ID.

    Slow path: unmapped track_ids (first sighting, or BoT-SORT re-detecting a
    player under a new ID) are scored against recently-lost players by predicted
    position -- last known velocity extrapolated over the frame gap -- normalized
    by the player's own bbox height, since a given pixel error means a lot for a
    tiny distant player and nothing for a large close one. A bbox size-ratio
    sanity gate and a team-color gate guard against merging two different,
    merely nearby, players.

    Three things make that slow path survive players converging, which is when a
    wrong merge is both most likely and most damaging (a swapped identity
    silently moves one player's distance and heatmap onto another's):

    * **Team gate.** Two players in different jerseys are never the same person,
      so a cross-team merge is refused outright regardless of how well the
      geometry lines up. Free, since jersey color is already sampled for team
      classification.

    * **Joint assignment, not greedy.** All unmapped detections in a cycle are
      matched to all lost candidates at once (Hungarian, via scipy), so the
      result doesn't depend on detection order and two detections can't compete
      for the same player. Taking each detection's own nearest match in turn
      lets the first one claim a player that a later detection fits far better,
      which is exactly the arrangement a crossing produces.

    * **Occlusion-aware state.** While a player's box overlaps another's -- or
      has swallowed one whose track just vanished into it -- its position is
      partly the *other* player's. Velocity is frozen at its last clean value
      and matching predicts forward from the last clean position, so the players
      that emerge from a merge are resolved by the motion they carried into it
      rather than by the merged box's meaningless drift. Occluded players are
      reported in `occluded_player_ids` so downstream analytics can drop those
      samples instead of trusting a contaminated position.

    The values below are a starting point (deliberately conservative -- start
    strict on merges, loosen only against measured false-splits), not tuned
    against real match footage yet.
    """

    MAX_LOST_FRAMES = 30  # ~1.25s at 24fps
    MAX_NORM_DIST = 3.0  # (predicted-position error) / bbox_height
    SIZE_RATIO_RANGE = (0.5, 2.0)
    _NO_MATCH = 1e6  # cost standing in for "gate failed"; never an accepted pairing
    _TEAM_BLOCKED = 2e6  # _NO_MATCH, but attributable to the team gate when counting

    def __init__(self):
        self.players: dict[int, PlayerState] = {}
        self.tracker_to_player: dict[int, int] = {}
        self.occluded_player_ids: set[int] = set()
        self._next_id = 1
        self.switch_log = []  # instrumentation: every track_id reconciliation
        # instrumentation: ordered player_ids each BoT-SORT track has belonged to.
        # A track that changes owner is an identity swap in progress, and one that
        # returns to an owner it already left cannot be anything else -- this is the
        # measurement switch_log can't give, since it counts merges without saying
        # whether they were right.
        self.track_owners: dict[int, list[int]] = {}
        self.team_blocked_merges = 0
        self.occluded_cycles = 0

    @property
    def total_players_minted(self) -> int:
        """Count of persistent player_ids ever created (active, lost, or retired)."""
        return self._next_id - 1

    def update(self, detections: list, frame_id: int, team_gate=None) -> list:
        """detections: (x1, y1, x2, y2, track_id) tuples from extract_boxes.

        Returns the same shape with track_id replaced by a persistent
        player_id. Untracked detections (track_id < 0) pass through as -1.
        """
        present_tracker_ids = {tid for *_, tid in detections if tid >= 0}

        # Mark vanished-this-frame players lost *before* reconciling new/unmapped
        # track_ids, so a same-cycle reacquisition (the 95 -> 159 case) sees the
        # just-vanished player as a candidate immediately, not one frame late.
        just_lost = []
        for player in self.players.values():
            if player.status == "active" and player.tracker_id not in present_tracker_ids:
                player.status = "lost"
                just_lost.append(player)

        contamination = self._occlusion_flags(detections, just_lost)

        results = [None] * len(detections)
        unmapped = []
        for i, (x1, y1, x2, y2, tracker_id) in enumerate(detections):
            if tracker_id < 0:
                results[i] = (x1, y1, x2, y2, -1)
                continue
            player_id = self.tracker_to_player.get(tracker_id)
            if player_id is None:
                unmapped.append(i)
                continue
            occluded = self._is_occluded(contamination[i], player_id)
            self._observe(player_id, tracker_id, (x1, y1, x2, y2), frame_id, occluded, team_gate)
            results[i] = (x1, y1, x2, y2, player_id)

        for i, player_id in self._reconcile_batch(detections, unmapped, frame_id, team_gate):
            x1, y1, x2, y2, tracker_id = detections[i]
            occluded = self._is_occluded(contamination[i], player_id)
            self._observe(player_id, tracker_id, (x1, y1, x2, y2), frame_id, occluded, team_gate)
            results[i] = (x1, y1, x2, y2, player_id)

        self._retire_stale(frame_id)
        self.occluded_player_ids = {
            p.player_id for p in self.players.values() if p.status == "active" and p.occluded
        }
        self.occluded_cycles += len(self.occluded_player_ids)
        return results

    def _occlusion_flags(self, detections: list, just_lost: list) -> list:
        """Per detection: which other players contaminate this box right now,
        as (overlaps a live detection, {player_ids swallowed by it}).

        Two ways that happens. The box still overlaps another live detection --
        two players contesting a ball, each box containing some of the other.
        Or the detector collapsed both players into one box, in which case there
        is no second detection to overlap; the tell is that a player's track
        vanished this cycle into a box that covers where it just was.
        """
        contamination = []
        for i, (x1, y1, x2, y2, _) in enumerate(detections):
            box = (x1, y1, x2, y2)
            overlaps_live = any(
                boxes_overlap(box, (ox1, oy1, ox2, oy2))
                for j, (ox1, oy1, ox2, oy2, _) in enumerate(detections)
                if j != i
            )
            swallowed = {p.player_id for p in just_lost if boxes_overlap(box, p.bbox)}
            contamination.append((overlaps_live, swallowed))
        return contamination

    @staticmethod
    def _is_occluded(contamination, player_id: int) -> bool:
        """A box sitting on top of the track it is itself continuing is just a
        reacquisition, not an occlusion -- only *another* player's presence in
        the box contaminates it."""
        overlaps_live, swallowed = contamination
        return overlaps_live or bool(swallowed - {player_id})

    def _reconcile_batch(self, detections: list, unmapped: list, frame_id: int, team_gate) -> list:
        """Assign every unmapped detection in this cycle to a recently-lost
        player, or mint a new player_id where nothing fits. Solved as one
        assignment problem so the outcome doesn't depend on detection order and
        no two detections can claim the same player."""
        if not unmapped:
            return []

        candidates = [
            p
            for p in self.players.values()
            if p.status == "lost" and 0 < frame_id - p.last_seen_frame <= self.MAX_LOST_FRAMES
        ]
        assignments = {}
        if candidates:
            costs = np.array(
                [
                    [
                        self._match_cost(detections[i], candidate, frame_id, team_gate)
                        for candidate in candidates
                    ]
                    for i in unmapped
                ]
            )
            # Count a refusal once per detection, not once per candidate it was
            # scored against -- a detection rejected against ten candidates is
            # one merge refused, not ten.
            self.team_blocked_merges += int(
                sum(
                    1
                    for row in costs
                    if (row == self._TEAM_BLOCKED).any() and (row < self._NO_MATCH).sum() == 0
                )
            )
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols, strict=True):
                if costs[row][col] >= self._NO_MATCH:
                    continue
                assignments[unmapped[row]] = (candidates[col], costs[row][col])

        resolved = []
        for i in unmapped:
            tracker_id = detections[i][4]
            matched = assignments.get(i)
            if matched is None:
                resolved.append((i, self._mint(tracker_id)))
                continue
            player, cost = matched
            self.switch_log.append(
                {
                    "frame": frame_id,
                    "player_id": player.player_id,
                    "old_tracker_id": player.tracker_id,
                    "new_tracker_id": tracker_id,
                    "norm_dist": round(float(cost), 2),
                }
            )
            self.tracker_to_player.pop(player.tracker_id, None)
            self.tracker_to_player[tracker_id] = player.player_id
            resolved.append((i, player.player_id))
        return resolved

    def _match_cost(self, detection, player: PlayerState, frame_id: int, team_gate) -> float:
        """Normalized predicted-position error, or _NO_MATCH if any gate rejects
        the pairing outright."""
        x1, y1, x2, y2, _ = detection
        px1, py1, px2, py2 = player.clean_bbox or player.bbox
        p_height = max(1, py2 - py1)

        size_ratio = max(1, y2 - y1) / p_height
        if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
            return self._NO_MATCH

        if team_gate is not None and player.team is not None:
            team = team_gate.of_box((x1, y1, x2, y2))
            if team is not None and team != player.team:
                return self._TEAM_BLOCKED

        # Predict from the last *clean* sighting: once a player's box has merged
        # with someone else's, both its position and the time since it was last
        # trustworthy are measured from before the merge.
        gap = frame_id - (player.clean_frame or player.last_seen_frame)
        if gap <= 0 or gap > self.MAX_LOST_FRAMES:
            return self._NO_MATCH
        pred_cx = (px1 + px2) / 2 + player.vx * gap
        pred_cy = (py1 + py2) / 2 + player.vy * gap
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        norm_dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5 / p_height
        return norm_dist if norm_dist <= self.MAX_NORM_DIST else self._NO_MATCH

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
        occluded: bool,
        team_gate=None,
    ) -> None:
        existing = self.players.get(player_id)
        vx, vy = 0.0, 0.0
        clean_bbox, clean_frame = (None, None) if occluded else (bbox, frame_id)
        team = None
        if existing is not None:
            team = existing.team
            if occluded:
                # The box is partly someone else's: neither its position nor the
                # jump to it says anything about this player's own motion.
                vx, vy = existing.vx, existing.vy
                clean_bbox, clean_frame = existing.clean_bbox, existing.clean_frame
            elif existing.clean_frame is not None:
                gap = max(1, frame_id - existing.clean_frame)
                pcx = (existing.clean_bbox[0] + existing.clean_bbox[2]) / 2
                pcy = (existing.clean_bbox[1] + existing.clean_bbox[3]) / 2
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                vx, vy = (cx - pcx) / gap, (cy - pcy) / gap
        if team_gate is not None:
            known = team_gate.of_player(player_id)
            team = known if known is not None else team

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
            team=team,
            occluded=occluded,
            clean_bbox=clean_bbox or bbox,
            clean_frame=clean_frame if clean_frame is not None else frame_id,
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
        """Tracks that changed hands between player_ids -- the direct swap count."""
        return {t: owners for t, owners in self.track_owners.items() if len(owners) > 1}

    def summary(self) -> str:
        contested = self.contested_tracks
        lines = [
            f"Persistent player IDs minted: {self.total_players_minted}",
            f"track_id -> player_id reconciliations (switches absorbed): {len(self.switch_log)}",
            f"Reconciliations refused outright on a team-color mismatch: "
            f"{self.team_blocked_merges}",
            f"Player-cycles spent occluded (position unreliable): {self.occluded_cycles}",
            f"Tracks that changed owner (identity swaps): {len(contested)}"
            + (f" {contested}" if contested else ""),
        ]
        for event in self.switch_log[:20]:
            lines.append(
                f"  frame {event['frame']}: track {event['old_tracker_id']} -> "
                f"{event['new_tracker_id']} kept as player {event['player_id']} "
                f"(norm_dist={event['norm_dist']})"
            )
        if len(self.switch_log) > 20:
            lines.append(f"  ... and {len(self.switch_log) - 20} more")
        return "\n".join(lines)
