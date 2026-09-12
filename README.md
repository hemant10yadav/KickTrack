# KickTrack

A computer-vision tool that tracks players in football match footage and turns it into player and team performance analytics.

KickTrack detects and tracks every player on the pitch from fixed/elevated camera footage in near real time, overlaying live markers on their positions. It's built to grow into a full analytics tool covering:

- **Player-level insights**: movement heatmaps, distance covered, speed/sprints, zone occupancy
- **Match-level insights**: team formations, possession, passing networks, tactical shape

## Current status

Player detection + tracking with live markers is working end-to-end. See [`docs/PLAN.md`](docs/PLAN.md) for the full milestone checklist and what's next.

## Tech stack

- **Detection**: [Ultralytics YOLO](https://docs.ultralytics.com/) (`yolov8m`)
- **Tracking**: BoT-SORT with camera motion compensation (handles panning/zooming footage)
- **CV/video**: OpenCV
- **Package management**: [uv](https://docs.astral.sh/uv/)
- **Linting/formatting**: [Ruff](https://docs.astral.sh/ruff/) via [prek](https://github.com/j178/prek) pre-commit hooks
- **Backend (scaffolded)**: FastAPI

## Setup

```bash
uv sync
```

To enable the pre-commit hooks (lint + format on every commit):

```bash
uv run prek install
```

## Usage

Run the tracker on a video (defaults to `data/videos/sample.mp4` if no path given):

```bash
uv run python scripts/track_players.py data/videos/match_1.mp4
```

This opens a live window playing the video with a marker + ID on each detected player. Press `q` to quit.

## Project structure

```
scripts/track_players.py   # tracking pipeline (PlayerTracker, InferenceWorker, MarkerRenderer, FramePacer)
scripts/botsort_custom.yaml # tracker tuning (camera motion compensation, track buffer)
data/videos/                # test footage
docs/PLAN.md                 # milestone checklist
app/                         # FastAPI skeleton (not yet the active focus)
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
```

See [`CLAUDE.md`](CLAUDE.md) for architecture decisions and project-specific context.
