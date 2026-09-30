# KickTrack

A computer-vision tool that tracks players in football match footage and turns it into player and team performance analytics.

KickTrack detects and tracks every player on the pitch from fixed/elevated camera footage in near real time, overlaying live markers on their positions. It's built to grow into a full analytics tool covering:

- **Player-level insights**: movement heatmaps, distance covered, speed/sprints, zone occupancy
- **Match-level insights**: team formations, possession, passing networks, tactical shape

## Current status

Player tracking, team classification by kit colour, distance covered and heatmaps, ball possession and team pass counts work end-to-end, live, with a web UI to switch overlays while a video plays. See [`docs/PLAN.md`](docs/PLAN.md) for the full milestone checklist and what's next.

## Demo

Tracked match clips: the distance each player has run, pitch lines from the
calibrated homography, and a live minimap. Click a clip to play it, or watch them
all on the **[demo page](https://hemant10yadav.github.io/KickTrack/)**.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://hemant10yadav.github.io/KickTrack/demos/match_5.mp4"><img src="https://hemant10yadav.github.io/KickTrack/demos/match_5.jpg" alt="EPL demo, Man City v Tottenham" width="100%"></a>
      <br><b>EPL</b> · Man City v Tottenham
    </td>
    <td align="center" width="50%">
      <a href="https://hemant10yadav.github.io/KickTrack/demos/match_6.mp4"><img src="https://hemant10yadav.github.io/KickTrack/demos/match_6.jpg" alt="La Liga demo, Real Madrid v Barcelona" width="100%"></a>
      <br><b>La Liga</b> · Real Madrid v Barcelona
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://hemant10yadav.github.io/KickTrack/demos/match_7.mp4"><img src="https://hemant10yadav.github.io/KickTrack/demos/match_7.jpg" alt="Copa del Rey demo, Barcelona v Real Madrid" width="100%"></a>
      <br><b>Copa del Rey</b> · Barcelona v Real Madrid
    </td>
    <td align="center" width="50%">
      <a href="https://hemant10yadav.github.io/KickTrack/demos/match_4.mp4"><img src="https://hemant10yadav.github.io/KickTrack/demos/match_4.jpg" alt="EPL demo, Tottenham v Watford" width="100%"></a>
      <br><b>EPL</b> · Tottenham v Watford
    </td>
  </tr>
</table>

Rendered with `./demo.sh` and published to the `gh-pages` branch by `./demo.sh --publish`.

## Tech stack

- **Detection**: [Ultralytics YOLO](https://docs.ultralytics.com/) (`yolo26s`, CoreML on the Neural Engine)
- **Tracking**: BoT-SORT with camera motion compensation (handles panning/zooming footage)
- **CV/video**: OpenCV
- **Package management**: [uv](https://docs.astral.sh/uv/)
- **Linting/formatting**: [Ruff](https://docs.astral.sh/ruff/) via [prek](https://github.com/j178/prek) pre-commit hooks

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
uv run python -m scripts.track_players data/videos/match_5.mp4
```

This opens a live window playing the video with a marker + ID on each detected player. Press `q` to quit.

Or use the web UI: pick a video (from `data/videos/`, or any file through the
browser's file chooser), watch it tracked live, and switch overlays (pitch
lines, player pins and IDs, distance run, ball rings, pass panel, minimap, FPS)
on and off while it plays:

```bash
uv run python -m web                 # then open http://127.0.0.1:8000
uv run python -m web --port 9000     # another port; --host 0.0.0.0 for the LAN
```

## Project structure

```
scripts/track_players.py   # CLI entrypoint (parse_args, main)
scripts/pipeline.py        # tracking pipeline (PlayerTracker, InferenceWorker, detection, video source)
scripts/player.py          # TeamClassifier, PlayerIdentityManager, StateManager
scripts/display.py         # Overlays, MarkerRenderer, FramePacer, motion/fade/stats
web/app.py                 # FastAPI web UI: video list/upload, tracking sessions, overlay API
web/stream.py              # annotated frames to the browser as MJPEG (FrameBroadcaster)
web/static/index.html      # the page: videos, live video, overlay checkboxes
scripts/botsort_custom.yaml # tracker tuning (camera motion compensation, track buffer)
data/videos/                # test footage
docs/PLAN.md                 # milestone checklist
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
```

See [`CLAUDE.md`](CLAUDE.md) for architecture decisions and project-specific context.
