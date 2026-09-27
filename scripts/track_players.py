import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from scripts import pipeline
from scripts.pipeline import PlayerTracker, is_stream_source, resolve_video_source

DEFAULT_VIDEO = "data/videos/sample.mp4"
DEFAULT_DISPLAY_DELAY_MS = 100
DEFAULT_VIEWER = "ffplay" if shutil.which("ffplay") else "opencv"
OUTPUT_DIR = Path("output")


def parse_imgsz(raw: str) -> int | tuple[int, int]:
    """Accepts a single size ("1280", square) or "H,W" (rectangular, must match
    the shape the .mlpackage was exported with)."""
    if "," in raw:
        h, w = raw.split(",")
        return (int(h), int(w))
    return int(raw)


def parse_args():
    parser = argparse.ArgumentParser(description="Track players in a football video")
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=(
            f"Path to a local video file, a stream URL (rtmp://, rtsp://, http(s)://), "
            f"or a webcam device index (e.g. 0) (default: {DEFAULT_VIDEO})"
        ),
    )
    parser.add_argument(
        "--model",
        default=pipeline.MODEL_NAME,
        help=f"YOLO model (default: {pipeline.MODEL_NAME})",
    )
    parser.add_argument(
        "--imgsz",
        type=parse_imgsz,
        default=pipeline.INFERENCE_IMGSZ,
        help=f"Inference size: single int for square, or 'H,W' for rectangular "
        f"(default: {pipeline.INFERENCE_IMGSZ[0]},{pipeline.INFERENCE_IMGSZ[1]}) — "
        f"must match the shape the .mlpackage was exported with",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="",
        default=None,
        help="Write annotated frames to this video file, or stream them live to a URL "
        "(rtmp://, srt://, udp://, rtsp://), H.264-encoded on the hardware encoder via "
        "ffmpeg, instead of (or alongside) displaying them. It runs headless (no "
        "window) unless --show is also passed, and always at the source frame rate, so "
        "the file shows what the live window would. A bare file name is saved under "
        "output/; with no value, output/<video name>.mp4. A file keeps the source's audio.",
    )
    parser.add_argument(
        "--viewer",
        choices=["ffplay", "opencv"],
        default=DEFAULT_VIEWER,
        help="What draws the live window: ffplay (GPU, its own process; holds the source "
        "frame rate) or an OpenCV window, whose repaint costs ~14-18ms per frame on macOS "
        f"and caps playback near 40fps (default: {DEFAULT_VIEWER})",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=pipeline.DISPLAY_WIDTH,
        help=f"Shrink wider sources (e.g. 4K) to this width right after decode; detection, "
        f"drawing, the window and --output all run at it (default: {pipeline.DISPLAY_WIDTH})",
    )
    parser.add_argument(
        "--display-delay-ms",
        type=float,
        default=DEFAULT_DISPLAY_DELAY_MS,
        help="Show each frame this much later than it is read, so its markers come from "
        "detections of that same frame instead of trailing the players by an inference "
        f"cycle. 0 shows frames as read (default: {DEFAULT_DISPLAY_DELAY_MS:g})",
    )
    parser.add_argument(
        "--analytics-dir",
        default=None,
        help="Write per-player/team heatmap PNGs and stats.json (distance covered, "
        "top speed, time tracked) to this directory at the end of the run.",
    )
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Draw the team-coloured pin and ID label above each player, and the rings on "
        "the ball and its holder. Off by default: only the distance each player has run "
        "and the pass panel are shown.",
    )
    parser.add_argument(
        "--hide-passes",
        action="store_true",
        help="Leave out the pass panel (the passes are still counted and written by "
        "--analytics-dir).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a live window. Implied when --output is not given.",
    )
    return parser.parse_args()


def resolve_output(output: str | None, video_source: str | int) -> str | None:
    """A bare file name (or none) goes under OUTPUT_DIR; a path or URL is kept."""
    if output is None or is_stream_source(output):
        return output
    if not output:
        stem = Path(video_source).stem if isinstance(video_source, str) else "camera"
        output = f"{stem}.mp4"
    path = Path(output)
    if path.parent == Path("."):
        path = OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def main():
    args = parse_args()
    pipeline.MODEL_NAME = args.model
    pipeline.INFERENCE_IMGSZ = args.imgsz

    video_source = resolve_video_source(args.video)
    if not is_stream_source(video_source) and not Path(video_source).exists():
        print(f"Video not found: {video_source}")
        print("Place a fixed-camera football video there, or pass a path as an argument.")
        sys.exit(1)

    args.output = resolve_output(args.output, video_source)
    model = YOLO(pipeline.MODEL_NAME)
    show_window = args.show or args.output is None
    PlayerTracker(
        video_source,
        model,
        show_window=show_window,
        output_path=args.output,
        # Always paced: the inference and calibration workers only take the
        # newest frame, so an unpaced file run read match_6 3x faster than
        # they could follow -- calibration came late and passes went uncounted.
        realtime=True,
        analytics_dir=args.analytics_dir,
        display_width=args.display_width,
        display_delay_ms=args.display_delay_ms,
        viewer=args.viewer,
        show_markers=args.show_markers,
        show_passes=not args.hide_passes,
    ).run()


if __name__ == "__main__":
    main()
