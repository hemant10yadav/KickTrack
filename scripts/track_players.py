import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from scripts import pipeline
from scripts.pipeline import PlayerTracker, is_stream_source, resolve_video_source

DEFAULT_VIDEO = "data/videos/sample.mp4"


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
        help="Write annotated frames to this video file instead of (or alongside) "
        "displaying them. Runs headless (no window, no real-time pacing) unless "
        "--show is also passed.",
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
    PlayerTracker(
        video_source,
        model,
        show_window=show_window,
        output_path=args.output,
        realtime=show_window,
    ).run()


if __name__ == "__main__":
    main()
