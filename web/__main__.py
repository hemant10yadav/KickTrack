"""uv run python -m web            # then open http://127.0.0.1:8000"""

import argparse

import uvicorn

from scripts import pipeline
from web.app import create_app


def main():
    parser = argparse.ArgumentParser(description="KickTrack web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model", default=pipeline.MODEL_NAME, help=f"(default: {pipeline.MODEL_NAME})"
    )
    args = parser.parse_args()
    pipeline.MODEL_NAME = args.model
    print(f"KickTrack web UI on http://{args.host}:{args.port}")
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="warning",
        # uvicorn waits for open responses before the app's shutdown, and the
        # video stream only ends in that shutdown (it closes the broadcaster):
        # without a timeout an open page kept Ctrl-C and SIGTERM from exiting.
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
