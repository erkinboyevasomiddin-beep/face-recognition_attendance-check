from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from recognition import config
from recognition.app.recognizer import RecognitionResult

LOGGER = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    person_id: str | None
    label: str
    class_name: str | None
    consecutive_hits: int
    misses: int
    missing_frames: int
    confirmed: bool
    last_seen: float
    last_similarity: float


@dataclass(frozen=True)
class MultiFaceTrackerOutput:
    display_results: list[RecognitionResult]
    confirmed_results: list[RecognitionResult]
    active_track_count: int


class MultiFaceTracker:
    """Maintain lightweight per-face recognition state for crowded webcam scenes."""

    def __init__(
        self,
        display_min_similarity: float = config.RECOGNITION_DISPLAY_MIN_SIMILARITY,
        confirm_min_similarity: float = config.RECOGNITION_CONFIRM_MIN_SIMILARITY,
        required_consecutive_hits: int = config.RECOGNITION_REQUIRED_CONSECUTIVE_HITS,
        max_misses_before_reset: int = config.RECOGNITION_MAX_MISSES_BEFORE_RESET,
        match_distance_threshold: float = config.TRACK_MATCH_DISTANCE_THRESHOLD,
        max_missing_frames: int = config.TRACK_MAX_MISSING_FRAMES,
        max_age_seconds: float = config.TRACK_MAX_AGE_SECONDS,
        max_active_faces: int = config.MAX_ACTIVE_FACES,
    ) -> None:
        self.display_min_similarity = float(display_min_similarity)
        self.confirm_min_similarity = float(confirm_min_similarity)
        self.required_consecutive_hits = max(1, int(required_consecutive_hits))
        self.max_misses_before_reset = max(0, int(max_misses_before_reset))
        self.match_distance_threshold = max(1.0, float(match_distance_threshold))
        self.match_distance_threshold_sq = (
            self.match_distance_threshold * self.match_distance_threshold
        )
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.max_active_faces = max(1, int(max_active_faces))
        self._tracks: list[FaceTrack] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        raw_results: list[RecognitionResult],
        timestamp: float | None = None,
    ) -> MultiFaceTrackerOutput:
        now = time.perf_counter() if timestamp is None else timestamp
        self._drop_stale_tracks(now)

        limited_results = self._limit_results(raw_results)
        matched_track_indices: set[int] = set()
        display_results: list[RecognitionResult] = []
        confirmed_results: list[RecognitionResult] = []

        for result in limited_results:
            track_index = self._match_track_index(result, matched_track_indices)
            if track_index is None:
                track = self._create_track(result, now)
                self._tracks.append(track)
                track_index = len(self._tracks) - 1
            matched_track_indices.add(track_index)

            track = self._tracks[track_index]
            display_result, newly_confirmed = self._update_track(track, result, now)
            display_results.append(display_result)
            if newly_confirmed:
                confirmed_results.append(display_result)

        self._age_unmatched_tracks(matched_track_indices, now)

        return MultiFaceTrackerOutput(
            display_results=display_results,
            confirmed_results=confirmed_results,
            active_track_count=len(self._tracks),
        )

    def _limit_results(self, results: list[RecognitionResult]) -> list[RecognitionResult]:
        if len(results) <= self.max_active_faces:
            return list(results)

        limited = sorted(results, key=self._result_priority, reverse=True)[: self.max_active_faces]
        LOGGER.debug(
            "Limiting crowded scene from %d faces to %d active faces.",
            len(results),
            self.max_active_faces,
        )
        return limited

    def _result_priority(self, result: RecognitionResult) -> tuple[float, float]:
        return (self._bbox_area(result.bbox), result.detection_score)

    def _match_track_index(
        self,
        result: RecognitionResult,
        matched_track_indices: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_sort_key: tuple[int, float] | None = None

        for index, track in enumerate(self._tracks):
            if index in matched_track_indices:
                continue

            distance_sq = self._bbox_center_distance_sq(result.bbox, track.bbox)
            if distance_sq > self.match_distance_threshold_sq:
                continue

            label_mismatch = 0 if track.person_id == result.person_id else 1
            sort_key = (label_mismatch, distance_sq)
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_index = index

        return best_index

    def _create_track(self, result: RecognitionResult, now: float) -> FaceTrack:
        track = FaceTrack(
            track_id=self._next_track_id,
            bbox=result.bbox,
            person_id=None,
            label=config.UNKNOWN_LABEL,
            class_name=None,
            consecutive_hits=0,
            misses=0,
            missing_frames=0,
            confirmed=False,
            last_seen=now,
            last_similarity=result.similarity,
        )
        self._next_track_id += 1
        return track

    def _update_track(
        self,
        track: FaceTrack,
        result: RecognitionResult,
        now: float,
    ) -> tuple[RecognitionResult, bool]:
        track.bbox = result.bbox
        track.last_seen = now
        track.missing_frames = 0

        is_displayable = self._is_displayable(result)
        is_confirmable = is_displayable and result.similarity >= self.confirm_min_similarity

        if not is_displayable:
            track.misses += 1
            if track.misses > self.max_misses_before_reset:
                if track.confirmed or track.label != config.UNKNOWN_LABEL:
                    LOGGER.debug(
                        "Resetting track %d due to weak recognition.",
                        track.track_id,
                    )
                track.label = config.UNKNOWN_LABEL
                track.person_id = None
                track.class_name = None
                track.consecutive_hits = 0
                track.confirmed = False
                track.last_similarity = result.similarity
                return self._build_unknown_result(result), False

            if track.label != config.UNKNOWN_LABEL:
                return self._build_display_result(result, track, use_cached_similarity=True), False
            return self._build_unknown_result(result), False

        track.last_similarity = result.similarity
        track.misses = 0

        if track.person_id != result.person_id:
            if track.label != config.UNKNOWN_LABEL:
                LOGGER.debug(
                    "Track %d label changed from %s to %s.",
                    track.track_id,
                    track.label,
                    result.label,
                )
            track.label = result.label
            track.person_id = result.person_id
            track.class_name = result.class_name
            track.confirmed = False
            track.consecutive_hits = 1 if is_confirmable else 0
        elif track.confirmed:
            if is_confirmable:
                track.consecutive_hits = max(track.consecutive_hits, self.required_consecutive_hits)
        elif is_confirmable:
            track.consecutive_hits += 1
        else:
            track.consecutive_hits = max(0, track.consecutive_hits - 1)

        newly_confirmed = False
        if not track.confirmed and track.consecutive_hits >= self.required_consecutive_hits:
            track.confirmed = True
            newly_confirmed = True
            LOGGER.info(
                "Stable recognition confirmed | track=%d label=%s similarity=%.3f",
                track.track_id,
                track.label,
                result.similarity,
            )

        return self._build_display_result(result, track), newly_confirmed

    def _age_unmatched_tracks(self, matched_track_indices: set[int], now: float) -> None:
        kept_tracks: list[FaceTrack] = []
        for index, track in enumerate(self._tracks):
            if index in matched_track_indices:
                kept_tracks.append(track)
                continue

            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                continue
            if now - track.last_seen > self.max_age_seconds:
                continue
            kept_tracks.append(track)

        self._tracks = kept_tracks

    def _drop_stale_tracks(self, now: float) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if (now - track.last_seen) <= self.max_age_seconds
            and track.missing_frames <= self.max_missing_frames
        ]

    def _is_displayable(self, result: RecognitionResult) -> bool:
        return result.is_known and result.similarity >= self.display_min_similarity

    def _build_display_result(
        self,
        result: RecognitionResult,
        track: FaceTrack,
        *,
        use_cached_similarity: bool = False,
    ) -> RecognitionResult:
        similarity = track.last_similarity if use_cached_similarity else result.similarity
        if track.confirmed:
            return replace(
                result,
                person_id=track.person_id,
                label=track.label,
                similarity=similarity,
                is_known=True,
                class_name=track.class_name,
                display_label=track.label,
                stability_state="confirmed",
            )

        return replace(
            result,
            person_id=track.person_id,
            label=track.label,
            similarity=similarity,
            is_known=track.label != config.UNKNOWN_LABEL,
            class_name=track.class_name,
            display_label=f"{track.label} ({track.consecutive_hits}/{self.required_consecutive_hits})",
            stability_state="tentative",
        )

    def _build_unknown_result(self, result: RecognitionResult) -> RecognitionResult:
        return replace(
            result,
            person_id=None,
            label=config.UNKNOWN_LABEL,
            is_known=False,
            class_name=None,
            display_label=config.UNKNOWN_LABEL,
            stability_state="unknown",
        )

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
        return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))

    @staticmethod
    def _bbox_center_distance_sq(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        ax = (bbox_a[0] + bbox_a[2]) / 2.0
        ay = (bbox_a[1] + bbox_a[3]) / 2.0
        bx = (bbox_b[0] + bbox_b[2]) / 2.0
        by = (bbox_b[1] + bbox_b[3]) / 2.0
        dx = ax - bx
        dy = ay - by
        return (dx * dx) + (dy * dy)
