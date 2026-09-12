# track_players.py module split — design

## Motivation

`scripts/track_players.py` has grown to 1264 lines / ~19 classes and
functions as Plan 1 & 2 landed. It's now hard to navigate, and Plan 3
(analytics) will add more code on top of it. Splitting it serves three
goals at once: maintainability (smaller, single-purpose files),
testability (each domain can be imported and unit-tested without
pulling in the whole tracker), and reuse (the FastAPI backend in
`app/` will eventually need to reuse pieces of this pipeline).

## Grouping principle

Group by **domain**, not by individual class: "is this about what a
player *is*?" → `player.py`. "Is this about what gets *shown on
screen*?" → `display.py`. This keeps the mental model simple — a
future contributor asking "where would X live" answers it from the
domain, not from memorizing a class list.

Verified via the actual code (not docstring mentions) that this
produces no import cycles.

## Modules

| File | Domain | Contents |
|---|---|---|
| `video_source.py` | video input | `resolve_video_source`, `is_stream_source`, `STREAM_SCHEME_RE` |
| `detection.py` | raw detection pipeline | `track()`, `extract_boxes()`, `PERSON_CLASS_ID`, `TRACKER_CONFIG`, `INFERENCE_IMGSZ`, `MODEL_NAME` |
| `inference_worker.py` | pipeline orchestration | `InferenceResult`, `WorkerStats`, `InferenceWorker` |
| `player.py` | everything "what is a player" | `TeamClassifier`, `extract_jersey_color`, `boxes_overlap`, team constants, `StateManager`, `PlayerState`, `PlayerIdentityManager` |
| `display.py` | everything "what gets shown on screen" | `MarkerRenderer`, `FpsOverlay`, `MotionExtrapolator`, `DisplaySmoother`, `FadeController`, `StalenessTracker`, `DisplayStats`, `FramePacer` |
| `player_tracker.py` | top-level orchestrator | `PlayerTracker` |
| `track_players.py` | CLI entrypoint | `parse_imgsz`, `parse_args`, `main`, `DEFAULT_VIDEO` |

### Import direction (no cycles)

```
track_players.py
    └── player_tracker.py
            ├── video_source.py
            ├── inference_worker.py
            │       ├── detection.py
            │       └── player.py
            └── display.py   (no dependency on player.py or inference_worker.py)
```

`display.py`'s `FadeController` references `MarkerRenderer.MARKER_ALPHA`
as a class constant — both classes already live in `display.py`, so
this is an intra-module reference, not a cross-module one.

`track_players.py` mutates `detection.MODEL_NAME` /
`detection.INFERENCE_IMGSZ` from parsed CLI args (currently done via
`global` inside the same file) — this becomes an explicit
`detection.MODEL_NAME = args.model` module-attribute assignment.

## Constants placement

Each constant moves with the code that uses it (no shared
`constants.py` — avoids an extra file for ~9 unrelated values):
- `PERSON_CLASS_ID`, `TRACKER_CONFIG`, `INFERENCE_IMGSZ`, `MODEL_NAME` → `detection.py`
- `NUM_TEAM_CLUSTERS`, `TEAM_FIT_AFTER_SAMPLES`, `UNCLASSIFIED_COLOR`, `JERSEY_RESAMPLE_INTERVAL` → `player.py`
- `STREAM_SCHEME_RE` → `video_source.py`
- `DEFAULT_VIDEO` → `track_players.py` (CLI-only)

## Testing

`tests/test_tracking_quality.py`, `tests/test_pipeline_fps.py`, and
`tests/test_player_identity.py` currently import `MODEL_NAME`,
`InferenceWorker`, `PlayerTracker`, `PlayerIdentityManager` from
`scripts.track_players`. These imports are updated to point at the new
modules directly (`scripts.detection`, `scripts.inference_worker`,
`scripts.player_tracker`, `scripts.player`) — no backwards-compat
re-export shim in `track_players.py`, per project convention.

No new tests are added by this refactor; it's a pure move. The full
existing test suite (fast + slow) must pass unchanged after the split,
confirming behavior is identical.

## Non-goals

- No behavior change. This is a pure code-motion refactor.
- No new abstractions, interfaces, or dependency injection beyond what
  already exists — just relocating existing classes/functions.
- Not touching `app/` in this change (reuse from there is a future
  motivation, not a task here).

## Work location

Done in a new git worktree on a new branch (`hy/refactor-tracker-modules`
or similar), isolated from the current `hy/improve-output-frames`
branch's in-progress work.
