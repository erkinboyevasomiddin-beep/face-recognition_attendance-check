from __future__ import annotations

from recognition.app.multi_face_tracker import MultiFaceTracker
from recognition.app.recognizer import RecognitionResult
from recognition.app.stable_recognition import StableRecognitionManager


def result(person_id="DEMO-1", similarity=0.9, bbox=(0, 0, 50, 50)):
    return RecognitionResult(
        bbox=bbox,
        person_id=person_id,
        label="Demo Person" if person_id else "Unknown",
        similarity=similarity,
        is_known=person_id is not None,
        detection_score=0.99,
        class_name="5-01" if person_id else None,
    )


def test_temporal_confirmation_miss_tolerance_and_reset():
    manager = StableRecognitionManager(
        display_min_similarity=0.6,
        confirm_min_similarity=0.8,
        required_consecutive_hits=3,
        max_misses_before_reset=1,
        stability_time_window=2,
    )
    assert not manager.process([result()], timestamp=0.0).confirmed_results
    assert not manager.process([result()], timestamp=0.1).confirmed_results
    confirmed = manager.process([result()], timestamp=0.2)
    assert len(confirmed.confirmed_results) == 1
    assert manager.process([], timestamp=0.3).display_results == []
    manager.process([], timestamp=0.4)
    restarted = manager.process([result()], timestamp=0.5)
    assert restarted.display_results[0].stability_state == "tentative"


def test_multi_face_tracker_state_transitions_and_capacity():
    tracker = MultiFaceTracker(
        required_consecutive_hits=2,
        max_misses_before_reset=0,
        max_active_faces=1,
    )
    first = tracker.update(
        [result(bbox=(0, 0, 80, 80)), result("DEMO-2", bbox=(100, 0, 120, 20))], timestamp=0
    )
    assert first.active_track_count == 1
    second = tracker.update([result(bbox=(2, 1, 82, 81))], timestamp=0.1)
    assert len(second.confirmed_results) == 1
    unknown = tracker.update([result(None, similarity=0.1, bbox=(2, 1, 82, 81))], timestamp=0.2)
    assert unknown.display_results[0].stability_state == "unknown"


def test_stable_manager_label_change_weakening_stale_and_position_matching():
    manager = StableRecognitionManager(
        display_min_similarity=0.6,
        confirm_min_similarity=0.8,
        required_consecutive_hits=2,
        max_misses_before_reset=1,
        stability_time_window=1,
        position_tolerance=20,
    )
    assert (
        manager.process([result()], timestamp=0).display_results[0].stability_state == "tentative"
    )
    changed = manager.process([result("DEMO-2")], timestamp=0.1)
    assert changed.display_results[0].person_id == "DEMO-2"
    assert changed.display_results[0].stability_state == "tentative"
    weak = manager.process([result("DEMO-2", similarity=0.65)], timestamp=0.2)
    assert weak.display_results[0].stability_state == "tentative"
    unknown = manager.process([result(None, similarity=0.2)], timestamp=0.3)
    assert unknown.display_results[0].stability_state == "unknown"
    far = manager.process([result(bbox=(200, 200, 250, 250))], timestamp=0.4)
    assert far.display_results[0].stability_state == "tentative"
    manager.process([], timestamp=2.0)
    assert manager._candidates == []
    manager.reset()


def test_tracker_expiry_reset_and_unmatched_tracks():
    tracker = MultiFaceTracker(
        required_consecutive_hits=1,
        max_misses_before_reset=1,
        max_active_faces=2,
        max_age_seconds=0.5,
        max_missing_frames=1,
    )
    confirmed = tracker.update([result()], timestamp=0)
    assert confirmed.confirmed_results
    tracker.update([], timestamp=0.1)
    expired = tracker.update([], timestamp=1.0)
    assert expired.active_track_count == 0
    tracker.reset()
