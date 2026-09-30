"""The web UI's server: pick a video from data/videos/ (or upload one from
anywhere on disk into it), watch it tracked live in the browser, and switch
overlays on and off while it plays. Started by `uv run python -m web` (see __main__.py).

Underneath is the same PlayerTracker the CLI (scripts.track_players) runs,
one video at a time on a background thread. Its annotated frames reach the
page as an MJPEG stream (FrameBroadcaster), and the checkboxes set the
Overlays it reads on every frame. See docs/PLAN.md Plan 3.6.
"""

import asyncio
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from scripts import pipeline
from scripts.display import Overlays
from scripts.pipeline import PlayerTracker, is_stream_source, resolve_video_source
from web.stream import MEDIA_TYPE, FrameBroadcaster, mjpeg

VIDEO_DIR = Path("data/videos")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
STATIC_DIR = Path(__file__).parent / "static"


class TrackingSession:
    """One video tracked on a background thread, from loading the model to
    the last frame. It is the frame sink PlayerTracker writes to: frames go
    on to the broadcaster, and the first one marks playback as started.

    state: loading -> playing -> finished | stopping -> stopped | error
    """

    def __init__(self, source: str | int, overlays: Overlays, broadcaster: FrameBroadcaster):
        self.source = source
        self.state = "loading"
        self.error: str | None = None
        self._overlays = overlays
        self._broadcaster = broadcaster
        self._tracker: PlayerTracker | None = None
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frame) -> None:
        if self.state == "loading":
            self.state = "playing"
        self._broadcaster.write(frame)

    def stop(self) -> None:
        """Returns at once; the tracker shuts its workers down on the session's
        own thread (join() waits for that). That can take 2s: a calibration
        process still loading its model can't take the stop sentinel and is
        killed when CalibrationWorker.stop()'s join times out."""
        if self.state in ("loading", "playing"):
            self.state = "stopping"
        self._stop_requested.set()
        if self._tracker is not None:
            self._tracker.stop()

    def join(self) -> None:
        self._thread.join()

    def status(self) -> dict:
        return {"state": self.state, "source": str(self.source), "error": self.error}

    def _run(self) -> None:
        try:
            # A fresh model per video: BoT-SORT's tracks live on the model
            # (persist=True) and each PlayerTracker installs its ball callback.
            model = YOLO(pipeline.MODEL_NAME)
            self._tracker = PlayerTracker(
                self.source,
                model,
                show_window=False,
                display_delay_ms=pipeline.DISPLAY_DELAY_MS,
                overlays=self._overlays,
                frame_sink=self,
            )
            if self._stop_requested.is_set():  # stop() came before the tracker existed
                self.state = "stopped"
                return
            self._tracker.run()
            self.state = "stopped" if self._stop_requested.is_set() else "finished"
        except Exception as error:
            traceback.print_exc()
            self.state = "error"
            self.error = str(error)


class SessionManager:
    """The one video being tracked (the Neural Engine runs one model at a
    time), and the overlays and broadcaster every video shares, so the
    checkboxes and the page's stream carry over from one video to the next.

    A replaced session winds down in the background while the next one loads:
    its display loop ends within a frame, so it can't draw into the new
    video's stream, which has no frames for the seconds its model takes to load.
    """

    def __init__(self):
        self.overlays = Overlays()
        self.broadcaster = FrameBroadcaster()
        self._session: TrackingSession | None = None
        self._retired: list[TrackingSession] = []  # stopped, maybe still winding down
        self._lock = threading.Lock()

    def play(self, source: str | int) -> dict:
        with self._lock:
            self._retire_current()
            self._session = TrackingSession(source, self.overlays, self.broadcaster)
            return self._session.status()

    def stop(self) -> dict:
        with self._lock:
            if self._session is not None:
                self._session.stop()
            return self.status()

    def status(self) -> dict:
        if self._session is None:
            return {"state": "idle", "source": None, "error": None}
        return self._session.status()

    def close(self) -> None:
        with self._lock:
            self._retire_current()
            for session in self._retired:
                session.join()
        self.broadcaster.close()

    def _retire_current(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._retired.append(self._session)
            self._session = None
        self._retired = [
            s for s in self._retired if s.state not in ("finished", "stopped", "error")
        ]


class PlayRequest(BaseModel):
    source: str  # a path, a stream URL or a webcam index, as the CLI takes it


def video_entry(path: Path) -> dict:
    return {"name": path.name, "path": str(path), "size": path.stat().st_size}


def list_videos(video_dir: Path) -> list[dict]:
    if not video_dir.is_dir():
        return []
    return [
        video_entry(path)
        for path in sorted(video_dir.iterdir())
        if path.suffix.lower() in VIDEO_SUFFIXES and not path.name.startswith(".")
    ]


def unused_path(path: Path) -> Path:
    """`path`, or "name (1).mp4", "name (2).mp4"... if taken: an upload never
    replaces a video already there."""
    candidate, n = path, 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
    return candidate


async def save_upload(request: Request, video_dir: Path, filename: str) -> Path:
    """Writes the request body to video_dir/filename (or an unused variant of
    it). The body is the raw file, not a multipart form, so it is written once,
    with no temporary copy in between. It goes to a hidden .part file first,
    renamed only once complete: an upload cut off half-way leaves nothing."""
    video_dir.mkdir(parents=True, exist_ok=True)
    partial = video_dir / f".{filename}.{uuid.uuid4().hex}.part"
    try:
        with partial.open("wb") as out:
            async for chunk in request.stream():
                # Off the event loop, which is also serving the video stream.
                await asyncio.to_thread(out.write, chunk)
        target = unused_path(video_dir / filename)
        partial.rename(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def overlay_states(overlays: Overlays) -> list[dict]:
    on = overlays.as_dict()
    return [
        {"name": name, "label": label, "on": on[name]} for name, label in overlays.labels().items()
    ]


def create_app(video_dir: Path = VIDEO_DIR) -> FastAPI:
    sessions = SessionManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await asyncio.to_thread(sessions.close)

    app = FastAPI(title="KickTrack", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/videos")
    def videos():
        return list_videos(video_dir)

    @app.post("/api/videos", status_code=201)
    async def upload(request: Request, name: str):
        """The file picked in the browser, sent as the raw body. A browser
        never tells the page where a picked file lives, so it can't be played
        in place: it is copied into video_dir, next to the others."""
        filename = Path(name).name  # no directories from the client
        if not filename or Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(415, f"Not a video file: {name}")
        return video_entry(await save_upload(request, video_dir, filename))

    @app.get("/api/overlays")
    def get_overlays():
        return overlay_states(sessions.overlays)

    @app.patch("/api/overlays")
    def set_overlays(changes: dict[str, bool]):
        try:
            sessions.overlays.update(changes)
        except KeyError as error:
            raise HTTPException(422, error.args[0]) from None
        return overlay_states(sessions.overlays)

    @app.get("/api/session")
    def session_status():
        return sessions.status()

    @app.post("/api/session")
    def play(request: PlayRequest):
        source = resolve_video_source(request.source.strip())
        if not is_stream_source(source) and not Path(source).is_file():
            raise HTTPException(404, f"Video not found: {source}")
        return sessions.play(source)

    @app.delete("/api/session")
    def stop():
        return sessions.stop()

    @app.get("/stream")
    async def stream():
        return StreamingResponse(mjpeg(sessions.broadcaster), media_type=MEDIA_TYPE)

    return app
