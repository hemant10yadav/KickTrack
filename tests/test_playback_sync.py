import numpy as np

from scripts.display import PlaybackDelay, ResultTimeline
from scripts.pipeline import FrameScaler, InferenceResult


def _result(frame_id, boxes):
    return InferenceResult(
        boxes=boxes,
        frame_id=frame_id,
        captured_at=None,
        previous_boxes=[],
        previous_captured_at=None,
        coasting_progress={},
    )


def test_frame_between_two_results_gets_interpolated_boxes():
    timeline = ResultTimeline()
    timeline.add(_result(10, [(100, 100, 120, 160, 1)]))
    timeline.add(_result(14, [(140, 100, 160, 160, 1)]))
    boxes, _, synced = timeline.boxes_at(11)
    assert synced
    assert boxes == [(110, 100, 130, 160, 1)]


def test_frame_with_its_own_result_gets_exactly_those_boxes():
    timeline = ResultTimeline()
    timeline.add(_result(10, [(100, 100, 120, 160, 1)]))
    timeline.add(_result(14, [(140, 100, 160, 160, 1)]))
    boxes, _, synced = timeline.boxes_at(14)
    assert synced and boxes == [(140, 100, 160, 160, 1)]


def test_player_new_in_the_later_result_is_shown_where_it_appeared():
    timeline = ResultTimeline()
    timeline.add(_result(10, [(100, 100, 120, 160, 1)]))
    timeline.add(_result(14, [(140, 100, 160, 160, 1), (500, 500, 520, 560, 2)]))
    boxes, _, _ = timeline.boxes_at(12)
    assert (500, 500, 520, 560, 2) in boxes


def test_frame_newer_than_every_result_is_extrapolated_in_video_time():
    timeline = ResultTimeline()
    timeline.add(_result(10, [(100, 100, 120, 160, 1)]))
    timeline.add(_result(12, [(110, 100, 130, 160, 1)]))  # 5px per frame
    boxes, _, synced = timeline.boxes_at(14)
    assert not synced
    assert boxes == [(120, 100, 140, 160, 1)]


def test_extrapolation_is_clamped():
    timeline = ResultTimeline(max_shift_px=30)
    timeline.add(_result(10, [(100, 100, 120, 160, 1)]))
    timeline.add(_result(11, [(150, 100, 170, 160, 1)]))
    boxes, _, _ = timeline.boxes_at(20)
    assert boxes == [(180, 100, 200, 160, 1)]


def test_repeated_result_is_added_once():
    timeline = ResultTimeline()
    result = _result(10, [(100, 100, 120, 160, 1)])
    timeline.add(result)
    timeline.add(result)
    assert len(timeline.results) == 1


def test_delay_holds_frames_back_and_flush_empties_it():
    playback = PlaybackDelay(delay_frames=2)
    shown = []
    for frame_id in range(1, 6):
        playback.push(f"frame{frame_id}", frame_id)
        if (item := playback.pop()) is not None:
            shown.append(item[1])
    assert shown == [1, 2, 3]
    while (item := playback.pop(flush=True)) is not None:
        shown.append(item[1])
    assert shown == [1, 2, 3, 4, 5]


def test_zero_delay_shows_each_frame_as_read():
    playback = PlaybackDelay(delay_frames=0)
    playback.push("frame", 1)
    assert playback.pop() == ("frame", 1)


def test_scaler_leaves_a_source_at_the_working_width_untouched():
    scaler = FrameScaler(1920, 1080, max_width=1920)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert not scaler.active
    assert scaler.to_working(frame) is frame


def test_scaler_shrinks_4k_and_rescales_its_homography():
    scaler = FrameScaler(3840, 2160, max_width=1920)
    assert scaler.size == (1920, 1080)
    assert scaler.to_working(np.zeros((2160, 3840, 3), dtype=np.uint8)).shape == (1080, 1920, 3)
    homography = np.array([[20.0, 1.0, 1900.0], [0.5, 15.0, 1100.0], [0.0001, 0.0002, 1.0]])
    pitch_point = np.array([10.0, -5.0, 1.0])
    source = homography @ pitch_point
    working = scaler.homography_to_working(homography) @ pitch_point
    assert np.allclose(working[:2] / working[2], source[:2] / source[2] / 2)
