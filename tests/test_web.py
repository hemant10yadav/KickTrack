"""The web UI (web/): the JPEG hand-off to the browser and the HTTP API.

Nothing here loads a model or tracks a video; playing one is checked with a
real run (docs/PLAN.md Plan 3.6)."""

import threading

import cv2
import numpy as np
import pytest

from web.stream import FrameBroadcaster

# --- FrameBroadcaster -------------------------------------------------------


@pytest.fixture
def broadcaster():
    broadcaster = FrameBroadcaster()
    yield broadcaster
    broadcaster.close()


def _frame(value: int) -> np.ndarray:
    return np.full((72, 128, 3), value, dtype="uint8")


def test_broadcaster_hands_out_the_written_frame_as_jpeg(broadcaster):
    broadcaster.write(_frame(200))
    seq, jpeg = broadcaster.next_jpeg(0, timeout=2.0)
    assert seq == 1
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (72, 128, 3)
    assert abs(int(decoded.mean()) - 200) <= 2


def test_broadcaster_waits_for_a_frame_newer_than_the_last_one_seen(broadcaster):
    broadcaster.write(_frame(10))
    seq, _ = broadcaster.next_jpeg(0, timeout=2.0)
    assert broadcaster.next_jpeg(seq, timeout=0.05) is None
    broadcaster.write(_frame(20))
    assert broadcaster.next_jpeg(seq, timeout=2.0)[0] > seq


def test_close_wakes_a_waiting_viewer(broadcaster):
    results = []
    viewer = threading.Thread(target=lambda: results.append(broadcaster.next_jpeg(0, 10.0)))
    viewer.start()
    broadcaster.close()
    viewer.join(timeout=2.0)
    assert not viewer.is_alive()
    assert results == [None]


# --- HTTP API ---------------------------------------------------------------

pytest.importorskip("httpx")  # TestClient's transport (installed with ultralytics)
from fastapi.testclient import TestClient  # noqa: E402

from web.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    (tmp_path / "match_b.mp4").touch()
    (tmp_path / "match_a.MOV").touch()
    (tmp_path / "notes.txt").touch()
    with TestClient(create_app(video_dir=tmp_path)) as client:
        yield client


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>KickTrack</title>" in response.text


def test_videos_lists_only_video_files_sorted(client, tmp_path):
    videos = client.get("/api/videos").json()
    assert [v["name"] for v in videos] == ["match_a.MOV", "match_b.mp4"]
    assert videos[1]["path"] == str(tmp_path / "match_b.mp4")


def test_overlays_round_trip(client):
    overlays = {o["name"]: o for o in client.get("/api/overlays").json()}
    assert overlays["markers"]["on"] is False
    assert overlays["markers"]["label"]

    response = client.patch("/api/overlays", json={"markers": True})
    assert response.status_code == 200
    assert {o["name"]: o["on"] for o in response.json()}["markers"] is True
    assert {o["name"]: o["on"] for o in client.get("/api/overlays").json()}["markers"] is True


def test_unknown_overlay_is_rejected(client):
    response = client.patch("/api/overlays", json={"heatmap": True})
    assert response.status_code == 422
    assert "heatmap" in response.json()["detail"]


def test_session_starts_idle(client):
    assert client.get("/api/session").json() == {"state": "idle", "source": None, "error": None}


def test_playing_a_missing_file_is_a_404(client, tmp_path):
    response = client.post("/api/session", json={"source": str(tmp_path / "nope.mp4")})
    assert response.status_code == 404
    assert client.get("/api/session").json()["state"] == "idle"


def test_stop_with_nothing_playing_is_harmless(client):
    assert client.delete("/api/session").json()["state"] == "idle"


def test_upload_saves_the_body_into_the_video_dir(client, tmp_path):
    response = client.post("/api/videos", params={"name": "cup_final.mp4"}, content=b"video bytes")
    assert response.status_code == 201
    assert response.json() == {
        "name": "cup_final.mp4",
        "path": str(tmp_path / "cup_final.mp4"),
        "size": 11,
    }
    assert (tmp_path / "cup_final.mp4").read_bytes() == b"video bytes"
    assert "cup_final.mp4" in [v["name"] for v in client.get("/api/videos").json()]


def test_upload_never_replaces_a_video_already_there(client, tmp_path):
    (tmp_path / "match_b.mp4").write_bytes(b"original")
    first = client.post("/api/videos", params={"name": "match_b.mp4"}, content=b"new one")
    second = client.post("/api/videos", params={"name": "match_b.mp4"}, content=b"and again")
    assert first.json()["name"] == "match_b (1).mp4"
    assert second.json()["name"] == "match_b (2).mp4"
    assert (tmp_path / "match_b.mp4").read_bytes() == b"original"


def test_upload_keeps_only_the_file_name(client, tmp_path):
    response = client.post("/api/videos", params={"name": "../../escape.mp4"}, content=b"x")
    assert response.json()["path"] == str(tmp_path / "escape.mp4")
    assert not (tmp_path.parent.parent / "escape.mp4").exists()


def test_upload_of_a_non_video_is_rejected(client, tmp_path):
    response = client.post("/api/videos", params={"name": "notes.pdf"}, content=b"x")
    assert response.status_code == 415
    assert not (tmp_path / "notes.pdf").exists()


def test_upload_leaves_no_partial_files(client, tmp_path):
    client.post("/api/videos", params={"name": "clip.mp4"}, content=b"x" * 100_000)
    assert not list(tmp_path.glob(".*.part"))
