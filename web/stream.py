"""Getting annotated frames from the tracker into the browser: the newest
frame as a JPEG (FrameBroadcaster) and the multipart stream an <img> plays
as live video (mjpeg)."""

import asyncio
import threading

import cv2
import numpy as np

BOUNDARY = "frame"
MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"


class FrameBroadcaster:
    """Hands the newest annotated frame, as a JPEG, to any number of web
    viewers (web/app.py). write() only swaps a reference, so the display
    thread never waits on an encode or on a viewer: one thread encodes the
    newest frame (~3.2ms at 1080p, measured on match_5) and a viewer that
    falls behind gets the newest JPEG next, never a backlog of old ones.
    It outlives any one PlayerTracker: every video played goes into it.
    """

    JPEG_QUALITY = 80  # ~190KB per 1080p match_5 frame: ~9MB/s at 50fps, nothing on localhost

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None  # newest frame not yet encoded
        self._jpeg = None
        self._seq = 0  # bumped per encoded frame
        self.closed = False
        self._thread = threading.Thread(target=self._encode, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        """The caller must not draw on `frame` afterwards."""
        with self._condition:
            self._pending = frame
            self._condition.notify_all()

    def next_jpeg(self, after_seq: int, timeout: float) -> tuple[int, bytes] | None:
        """(seq, jpeg) of the first frame encoded after `after_seq`, waiting up
        to `timeout` seconds for one; None if none came or once closed."""
        with self._condition:
            self._condition.wait_for(lambda: self.closed or self._seq > after_seq, timeout)
            if self.closed or self._seq <= after_seq:
                return None
            return self._seq, self._jpeg

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()
        self._thread.join()

    def _encode(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self.closed or self._pending is not None)
                if self.closed:
                    return
                frame, self._pending = self._pending, None
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            if not ok:
                continue
            with self._condition:
                self._jpeg = jpeg.tobytes()
                self._seq += 1
                self._condition.notify_all()


async def mjpeg(broadcaster: FrameBroadcaster):
    """Every new JPEG as one part of a multipart stream; a slow client just
    skips to the newest frame. Each part is followed by the next boundary at
    once: a browser only treats a part as complete when that boundary
    arrives, so a part left open (the last frame of a finished video) was
    never shown and kept the page loading."""
    yield f"--{BOUNDARY}\r\n".encode()
    seq = 0
    while not broadcaster.closed:
        frame = await asyncio.to_thread(broadcaster.next_jpeg, seq, 1.0)
        if frame is None:
            continue
        seq, jpeg = frame
        header = f"Content-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
        yield header.encode() + jpeg + f"\r\n--{BOUNDARY}\r\n".encode()
