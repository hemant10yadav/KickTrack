import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person"
TRACKER_CONFIG = str(Path(__file__).parent / "botsort_custom.yaml")
MODEL_NAME = "yolov8m.pt"
INFERENCE_IMGSZ = 1280
DEFAULT_VIDEO = "data/videos/sample.mp4"


def track(model: YOLO, frame):
    """Run detection + tracking on a single frame and return raw Ultralytics results."""
    return model.track(
        frame,
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker=TRACKER_CONFIG,
        device="mps",
        conf=0.15,
        imgsz=INFERENCE_IMGSZ,
        verbose=False,
    )[0]


def extract_boxes(results):
    """Convert Ultralytics results into a plain list of (x1, y1, x2, y2, track_id)."""
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        boxes.append((x1, y1, x2, y2, track_id))
    return boxes


class MarkerRenderer:
    """Draws player markers (triangle + ID label) onto a frame."""

    def __init__(self, color=(0, 255, 0), size=14):
        self.color = color
        self.size = size

    def draw(self, frame, boxes):
        for x1, y1, x2, y2, track_id in boxes:
            self._draw_triangle(frame, x1, y1, x2)
            self._draw_label(frame, x1, y1, track_id)

    def _draw_triangle(self, frame, x1, y1, x2):
        center_x = (x1 + x2) // 2
        top_y = y1 - 4
        points = np.array(
            [
                [center_x, top_y],
                [center_x - self.size, top_y - self.size * 2],
                [center_x + self.size, top_y - self.size * 2],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [points], self.color)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 0), thickness=1)

    def _draw_label(self, frame, x1, y1, track_id):
        cv2.putText(
            frame, f"ID {track_id}", (x1, y1 - 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2,
        )


class InferenceWorker:
    """Runs detection + tracking continuously in the background, always working on
    the most recently submitted frame. Frames arriving faster than inference finishes
    are dropped (never queued), so the worker never falls further and further behind.
    """

    def __init__(self, model: YOLO):
        self.model = model
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_boxes = []
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def submit(self, frame):
        with self.lock:
            self.latest_frame = frame

    def get_boxes(self):
        with self.lock:
            return self.latest_boxes

    def _run(self):
        while self.running:
            frame = self._take_pending_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            results = track(self.model, frame)
            boxes = extract_boxes(results)

            with self.lock:
                self.latest_boxes = boxes

    def _take_pending_frame(self):
        with self.lock:
            frame, self.latest_frame = self.latest_frame, None
            return frame


class FramePacer:
    """Waits out the remainder of a frame's time budget, accounting for time
    already spent this iteration, so processing cost doesn't compound into extra lag.
    """

    def __init__(self, fps: float):
        self.frame_budget_ms = 1000 / fps

    def wait(self, iteration_start: float) -> int:
        elapsed_ms = (time.perf_counter() - iteration_start) * 1000
        remaining_ms = max(1, int(self.frame_budget_ms - elapsed_ms))
        return cv2.waitKey(remaining_ms) & 0xFF


class PlayerTracker:
    """Plays a video live, overlaying player tracking markers.

    Detection/tracking runs on a background thread (see InferenceWorker) so
    display always paces at the video's real frame rate, independent of how
    long a single inference call takes.
    """

    WINDOW_NAME = "Football Tracker"
    QUIT_KEY = ord("q")

    def __init__(self, video_path: Path, model: YOLO):
        self.video_path = video_path
        self.model = model
        self.renderer = MarkerRenderer()
        self.frame_count = 0

    def run(self):
        cap = self._open_capture()
        pacer = FramePacer(fps=cap.get(cv2.CAP_PROP_FPS) or 25)
        self._warmup()

        worker = InferenceWorker(self.model).start()
        try:
            self._play(cap, worker, pacer)
        finally:
            worker.stop()
            cap.release()
            cv2.destroyAllWindows()
            print(f"Read {self.frame_count} frames from {self.video_path}")

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            sys.exit(1)
        return cap

    def _warmup(self):
        """Pay the one-time GPU kernel compilation cost (MPS) before playback starts."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        track(self.model, blank_frame)

    def _play(self, cap: cv2.VideoCapture, worker: InferenceWorker, pacer: FramePacer):
        while True:
            iteration_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1

            worker.submit(frame.copy())
            self.renderer.draw(frame, worker.get_boxes())
            cv2.imshow(self.WINDOW_NAME, frame)

            if pacer.wait(iteration_start) == self.QUIT_KEY:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Track players in a football video")
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=f"Path to the input video (default: {DEFAULT_VIDEO})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Place a fixed-camera football video there, or pass a path as an argument.")
        sys.exit(1)

    model = YOLO(MODEL_NAME)
    PlayerTracker(video_path, model).run()


if __name__ == "__main__":
    main()
