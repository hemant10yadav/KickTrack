from dataclasses import dataclass

import cv2
import numpy as np

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
    MAX_NORM_DIST = 3.0  # (predicted-position error) / bbox_height
    SIZE_RATIO_RANGE = (0.5, 2.0)

    LONG_LOST_MAX_FRAMES = 90  # ~3.75s at 24fps -- color-gated, not position-gated
    COLOR_MAX_DIST = 40.0  # max BGR L2 distance between smoothed jersey colors

    SWAP_MISMATCH_CYCLES = 3  # consecutive cycles of team disagreement before acting
    SWAP_PROXIMITY_RATIO = 3.0  # max center distance / bbox height to call it a crossing

    def __init__(self, classifier=None):
        self.players: dict[int, PlayerState] = {}
        self.tracker_to_player: dict[int, int] = {}
        self._next_id = 1
        self.switch_log = []  # instrumentation: every track_id reconciliation
        self.swap_log = []  # instrumentation: every crossing swap corrected
        self.classifier = classifier
        self._color_pending: dict[int, int] = {}  # player_id -> frame first minted

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

        results = []
        for x1, y1, x2, y2, tracker_id in detections:
            if tracker_id < 0:
                results.append((x1, y1, x2, y2, -1))
                continue
            bbox = (x1, y1, x2, y2)
            player_id = self.tracker_to_player.get(tracker_id)
            if player_id is None:
                player_id = self._reconcile(tracker_id, bbox, frame_id)
            self._observe(player_id, tracker_id, bbox, frame_id)
            results.append((x1, y1, x2, y2, player_id))

        if self.classifier is not None:
            self._check_swaps(frame_id)

        self._retire_stale(frame_id)
        return results

    def _reconcile(self, tracker_id: int, bbox: tuple, frame_id: int) -> int:
        """Find the recently-lost player this new track_id is probably continuing,
        or mint a new player_id if no candidate passes the gates."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        height = max(1, bbox[3] - bbox[1])

        best_id, best_norm_dist = None, None
        for player in self.players.values():
            if player.status != "lost":
                continue
            gap = frame_id - player.last_seen_frame
            if gap <= 0 or gap > self.MAX_LOST_FRAMES:
                continue
            px1, py1, px2, py2 = player.bbox
            p_height = max(1, py2 - py1)
            size_ratio = height / p_height
            if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
                continue
            pred_cx = (px1 + px2) / 2 + player.vx * gap
            pred_cy = (py1 + py2) / 2 + player.vy * gap
            norm_dist = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5 / p_height
            if norm_dist > self.MAX_NORM_DIST:
                continue
            if best_norm_dist is None or norm_dist < best_norm_dist:
                best_norm_dist = norm_dist
                best_id = player.player_id

        if best_id is not None:
            old_tracker_id = self.players[best_id].tracker_id
            self.switch_log.append(
                {
                    "frame": frame_id,
                    "player_id": best_id,
                    "old_tracker_id": old_tracker_id,
                    "new_tracker_id": tracker_id,
                    "norm_dist": round(best_norm_dist, 2),
                }
            )
            self.tracker_to_player.pop(old_tracker_id, None)
            self.tracker_to_player[tracker_id] = best_id
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self.tracker_to_player[tracker_id] = new_id
        if self.classifier is not None:
            self._color_pending[new_id] = frame_id
        return new_id

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
        for player_id, created_frame in self._color_pending.items():
            candidate = self.players.get(player_id)
            if candidate is None or frame_id - created_frame > self.LONG_LOST_MAX_FRAMES:
                resolved.append(player_id)
                continue
            color = self.classifier.track_colors.get(player_id)
            if color is None:
                continue  # no jersey-color sample yet -- keep waiting
            match_id = self._find_color_match(candidate, color, frame_id)
            if match_id is not None:
                self._merge_into(candidate, match_id, frame_id)
                resolved.append(player_id)
        for player_id in resolved:
            del self._color_pending[player_id]

    def _find_color_match(self, candidate: PlayerState, color, frame_id: int):
        best_id, best_dist = None, None
        for player in self.players.values():
            if player.status != "lost" or player.player_id == candidate.player_id:
                continue
            gap = frame_id - player.last_seen_frame
            # Short gaps are already handled by position matching in _reconcile;
            # this path only covers the window position matching gave up on.
            if gap <= self.MAX_LOST_FRAMES or gap > self.LONG_LOST_MAX_FRAMES:
                continue
            old_color = self.classifier.track_colors.get(player.player_id)
            if old_color is None:
                continue
            c_height = max(1, candidate.bbox[3] - candidate.bbox[1])
            p_height = max(1, player.bbox[3] - player.bbox[1])
            size_ratio = c_height / p_height
            if not (self.SIZE_RATIO_RANGE[0] <= size_ratio <= self.SIZE_RATIO_RANGE[1]):
                continue
            dist = float(
                np.linalg.norm(np.asarray(color, dtype=float) - np.asarray(old_color, dtype=float))
            )
            if dist > self.COLOR_MAX_DIST:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best_id = dist, player.player_id
        return best_id

    def _merge_into(self, candidate: PlayerState, target_player_id: int, frame_id: int) -> None:
        """Fold a color-pending candidate into the long-lost player it matched,
        so future frames report the old, established player_id instead."""
        self.tracker_to_player[candidate.tracker_id] = target_player_id
        target = self.players[target_player_id]
        target.tracker_id = candidate.tracker_id
        target.bbox = candidate.bbox
        target.vx, target.vy = candidate.vx, candidate.vy
        target.last_seen_frame = candidate.last_seen_frame
        target.status = candidate.status
        self.switch_log.append(
            {
                "frame": frame_id,
                "player_id": target_player_id,
                "old_tracker_id": target.tracker_id,
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
        """
        mismatched = []
        for player in self.players.values():
            if player.status != "active":
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
        self.tracker_to_player[player_a.tracker_id], self.tracker_to_player[player_b.tracker_id] = (
            player_b.player_id,
            player_a.player_id,
        )
        player_a.tracker_id, player_b.tracker_id = player_b.tracker_id, player_a.tracker_id
        player_a.team_mismatch_streak = 0
        player_b.team_mismatch_streak = 0
        self.swap_log.append(
            {"frame": frame_id, "player_a": player_a.player_id, "player_b": player_b.player_id}
        )

    def summary(self) -> str:
        lines = [
            f"Persistent player IDs minted: {self.total_players_minted}",
            f"track_id -> player_id reconciliations (switches absorbed): {len(self.switch_log)}",
            f"crossing-player ID swaps corrected: {len(self.swap_log)}",
        ]
        for event in self.switch_log[:20]:
            detail = event.get("note") or f"norm_dist={event.get('norm_dist')}"
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
        return "\n".join(lines)
