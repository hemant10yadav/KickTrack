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
        default=None,
        help="Write annotated frames to this video file, or stream them live to a URL "
        "(rtmp://, srt://, udp://, rtsp://), H.264-encoded on the hardware encoder via "
        "ffmpeg, instead of (or alongside) displaying them. A file runs headless (no "
        "window, no real-time pacing) unless --show is also passed; a stream is always "
        "paced at the source frame rate.",
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
        help="Draw the team-coloured pin and ID label above each player. Off by default: "
        "only the distance each player has run is shown.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a live window. Implied when --output is not given.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline.MODEL_NAME = args.model
    pipeline.INFERENCE_IMGSZ = args.imgsz

    video_source = resolve_video_source(args.video)
    if not is_stream_source(video_source) and not Path(video_source).exists():
        print(f"Video not found: {video_source}")
        print("Place a fixed-camera football video there, or pass a path as an argument.")
        sys.exit(1)

    model = YOLO(pipeline.MODEL_NAME)
    show_window = args.show or args.output is None
    streaming = args.output is not None and is_stream_source(args.output)
    PlayerTracker(
        video_source,
        model,
        show_window=show_window,
        output_path=args.output,
        realtime=show_window or streaming,
        analytics_dir=args.analytics_dir,
        display_width=args.display_width,
        display_delay_ms=args.display_delay_ms,
        viewer=args.viewer,
        show_markers=args.show_markers,
    ).run()


if __name__ == "__main__":
    main()
