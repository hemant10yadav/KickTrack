# KickTrack

KickTrack turns football match footage into player and team analytics. Point it at a
video from an elevated or stand-side camera and it tracks every player and the ball,
works out which team each player is on from their jersey colour, and measures
what happened: distance covered, top speed, heatmaps, possession and passes.

## Demo

<video src="https://github.com/hemant10yadav/KickTrack/releases/download/demo-latest/demo.mp4" controls width="720">
  Your browser doesn't support inline video — download it directly:
  https://github.com/hemant10yadav/KickTrack/releases/download/demo-latest/demo.mp4
</video>

## What you get

**While the video plays**

- A marker and a stable ID on every player, coloured by team
- Each player's running distance under their marker
- The ball ringed in yellow, plus a ring around the player who has it
- A live top-down minimap of the pitch showing where everyone and the ball are
- A panel with each team's completed passes. Teams are named by kit colour
  ("white", "sky blue"), not by number

**When the run ends** (with `--analytics-dir`)

| File | What's in it |
| --- | --- |
| `stats.json` | Per player: team, time tracked, distance covered (m), top speed (m/s) |
| `heatmap_team_<team>.png` | Where each team spent its time on the pitch |
| `heatmap_player_<id>.png` | One heatmap per player tracked for at least 2 seconds |
| `passes.json` | Per team: completed passes (short / long) and passes lost |

A summary of the same numbers is also printed to the terminal.

## Requirements

- A Mac with Apple Silicon. Detection runs on the Neural Engine through CoreML, so
  KickTrack does not run on Linux or Windows.
- Python 3.13 (not 3.14: `coremltools` doesn't work on 3.14 yet)
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`). Needed for `--output` and for
  the smooth `ffplay` viewer; without it KickTrack falls back to an OpenCV window.

## Installation

```bash
git clone https://github.com/hemant10yadav/KickTrack.git
cd KickTrack
uv sync
```

KickTrack needs two sets of model weights, which aren't in the repository:

**1. The player/ball detector.** Download `yolo26s.pt` from the
[ultralytics assets `v8.4.0` release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
into the project folder, then convert it to CoreML:

```bash
uv run --with "numpy==2.3.5" python -c "from ultralytics import YOLO; YOLO('yolo26s.pt').export(format='coreml', imgsz=(640, 1152), half=True)"
```

This creates `yolo26s.mlpackage`, which is what KickTrack loads.

**2. The pitch calibration model**, which maps the video onto real pitch coordinates
so distances and speeds come out in metres:

```bash
bash scripts/download_calibration_weights.sh
```

Both downloads come from GitHub releases and can be slow. A long wait usually isn't a
hang.

## Usage

Watch a match with live tracking:

```bash
uv run python -m scripts.track_players path/to/match.mp4
```

Press `q` to quit.

Save the analytics:

```bash
uv run python -m scripts.track_players path/to/match.mp4 --analytics-dir out/
```

Write the annotated video to a file (runs as fast as possible, no window):

```bash
uv run python -m scripts.track_players path/to/match.mp4 --output annotated.mp4
```

Add `--show` to watch it at the same time.

Stream the annotated video live:

```bash
uv run python -m scripts.track_players path/to/match.mp4 --output rtmp://your-server/live/key
```

The input can be a video file, a stream URL (`rtmp://`, `rtsp://`, `http(s)://`) or a
webcam index such as `0`.

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `--analytics-dir DIR` | off | Write `stats.json`, `passes.json` and heatmaps to `DIR` at the end |
| `--output PATH_OR_URL` | off | Save the annotated video to a file, or stream it to an `rtmp://`, `srt://`, `udp://` or `rtsp://` URL |
| `--show` | | Show the live window as well when using `--output` |
| `--viewer {ffplay,opencv}` | `ffplay` if installed | What draws the live window. `ffplay` keeps up with 50fps footage; the OpenCV window tops out around 40fps |
| `--display-width PX` | `1920` | Shrink larger sources (e.g. 4K) to this width. Keeps 4K playback smooth |
| `--display-delay-ms MS` | `100` | Show each frame slightly late so markers line up exactly with the players. `0` shows frames immediately, with markers trailing a little |
| `--model PATH` | `yolo26s.mlpackage` | Use a different detector |
| `--imgsz H,W` | `640,1152` | Detector input size. Must match the size the model was exported at |

## Footage that works best

- **An elevated view of the pitch** (a stand, a gantry or a tall mast) where players
  don't hide each other for long stretches.
- **Visible pitch markings.** Distances, speeds, heatmaps and the minimap all depend on
  finding the lines. Frames where the lines can't be found are left out of the numbers
  rather than guessed at.
- **Clearly different kits.** Teams are told apart by jersey colour alone.
- Some panning and zooming is fine; the tracker compensates for camera movement.

## Limitations

- **Calibration is weakest on low stand-side cameras.** The pitch calibration model was
  trained on broadcast footage. On a low camera close to the touchline it can fail to
  find the pitch at all, and then there are no metre-based stats for that stretch of
  video. Tracking and team colours still work.
- **Half-pitch framing.** When the camera follows the ball and shows only part of the
  pitch, players out of frame aren't tracked, so their distance and heatmap only cover
  the time they were on screen.
- **The ball is small and often blurred.** Possession and pass counts are only as good
  as the ball detection. Expect some missed or extra passes.
- **Player IDs aren't names.** An ID stays with the same player through occlusions and
  crossings as best it can, but IDs are numbers, not shirt numbers or names.
- Formations, team shape and passing networks are planned but not built yet. See
  [`docs/PLAN.md`](docs/PLAN.md) for the roadmap.

## Contributing

```bash
uv sync --group dev
uv run prek install          # lint + format on every commit
uv run pytest -m "not slow"  # fast tests
uv run pytest -m slow        # full pipeline against real footage (needs the models)
```

Architecture and design decisions are in [`CLAUDE.md`](CLAUDE.md), and the milestone
history is in [`docs/PLAN.md`](docs/PLAN.md).
