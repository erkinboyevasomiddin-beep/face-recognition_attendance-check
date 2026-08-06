from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from recognition import config
from recognition.app.recognizer import RecognitionResult

LOGGER = logging.getLogger(__name__)


@dataclass
class RecognitionCandidate:
    person_id: str | None
    label: str
    class_name: str | None
    bbox: tuple[int, int, int, int]
    consecutive_hits: int
    misses: int
    confirmed: bool
    last_seen: float
    last_similarity: float


@dataclass(frozen=True)
class StableRecognitionOutput:
    display_results: list[RecognitionResult]
    confirmed_results: list[RecognitionResult]


class StableRecognitionManager:
    """Track tentative and confirmed recognitions across multiple passes."""

    def __init__(
        self,
        display_min_similarity: float = config.RECOGNITION_DISPLAY_MIN_SIMILARITY,
        confirm_min_similarity: float = config.RECOGNITION_CONFIRM_MIN_SIMILARITY,
        required_consecutive_hits: int = config.RECOGNITION_REQUIRED_CONSECUTIVE_HITS,
        max_misses_before_reset: int = config.RECOGNITION_MAX_MISSES_BEFORE_RESET,
        stability_time_window: float = config.RECOGNITION_STABILITY_TIME_WINDOW,
        position_tolerance: float = config.RECOGNITION_POSITION_TOLERANCE,
    ) -> None:
        self.display_min_similarity = float(display_min_similarity)
        self.confirm_min_similarity = float(confirm_min_similarity)
        self.required_consecutive_hits = max(1, int(required_consecutive_hits))
        self.max_misses_before_reset = max(0, int(max_misses_before_reset))
        self.stability_time_window = max(0.1, float(stability_time_window))
        self.position_tolerance = max(1.0, float(position_tolerance))
        self.position_tolerance_sq = self.position_tolerance * self.position_tolerance
        self._candidates: list[RecognitionCandidate] = []

    def reset(self) -> None:
        self._candidates.clear()

    def process(
        self,
        raw_results: list[RecognitionResult],
        timestamp: float | None = None,
    ) -> StableRecognitionOutput:
        now = time.perf_counter() if timestamp is None else timestamp
        self._drop_stale_candidates(now)

        display_results: list[RecognitionResult] = []
        confirmed_results: list[RecognitionResult] = []
        matched_indices: set[int] = set()
        removed_indices: set[int] = set()

        for result in raw_results:
            candidate_index = self._match_candidate_index(result, matched_indices)
            is_displayable = self._is_displayable(result)
            is_confirmable = is_displayable and result.similarity >= self.confirm_min_similarity

            if candidate_index is None:
                if is_displayable:
                    candidate = RecognitionCandidate(
                        person_id=result.person_id,
                        label=result.label,
                        class_name=result.class_name,
                        bbox=result.bbox,
                        consecutive_hits=1 if is_confirmable else 0,
                        misses=0,
                        confirmed=False,
                        last_seen=now,
                        last_similarity=result.similarity,
                    )
                    self._candidates.append(candidate)
                    candidate_index = len(self._candidates) - 1
                    matched_indices.add(candidate_index)
                    LOGGER.debug(
                        "Tentative recognition started | label=%s similarity=%.3f hits=%d/%d",
                        candidate.label,
                        result.similarity,
                        candidate.consecutive_hits,
                        self.required_consecutive_hits,
                    )
                    if candidate.consecutive_hits >= self.required_consecutive_hits:
                        candidate.confirmed = True
                        LOGGER.info(
                            "Stable recognition confirmed | label=%s similarity=%.3f",
                            candidate.label,
                            result.similarity,
                        )
                        confirmed_results.append(result)
                    display_results.append(self._build_display_result(result, candidate))
                else:
                    display_results.append(self._build_unknown_result(result))
                continue

            matched_indices.add(candidate_index)
            candidate = self._candidates[candidate_index]

            if not is_displayable:
                if not self._register_miss(candidate, reason="weak_or_unknown"):
                    removed_indices.add(candidate_index)
                display_results.append(self._build_unknown_result(result))
                continue

            if candidate.person_id != result.person_id:
                LOGGER.debug(
                    "Recognition reset due to label change | previous=%s new=%s",
                    candidate.label,
                    result.label,
                )
                candidate.label = result.label
                candidate.person_id = result.person_id
                candidate.class_name = result.class_name
                candidate.bbox = result.bbox
                candidate.consecutive_hits = 1 if is_confirmable else 0
                candidate.misses = 0
                candidate.confirmed = False
                candidate.last_seen = now
                candidate.last_similarity = result.similarity
                LOGGER.debug(
                    "Tentative recognition started | label=%s similarity=%.3f hits=%d/%d",
                    candidate.label,
                    result.similarity,
                    candidate.consecutive_hits,
                    self.required_consecutive_hits,
                )
            else:
                candidate.bbox = result.bbox
                candidate.last_seen = now
                candidate.last_similarity = result.similarity
                candidate.misses = 0

                if candidate.confirmed:
                    if is_confirmable:
                        candidate.consecutive_hits = max(
                            candidate.consecutive_hits,
                            self.required_consecutive_hits,
                        )
                elif is_confirmable:
                    candidate.consecutive_hits += 1
                    LOGGER.debug(
                        "Stable recognition hit | label=%s hits=%d/%d similarity=%.3f",
                        candidate.label,
                        candidate.consecutive_hits,
                        self.required_consecutive_hits,
                        result.similarity,
                    )
                else:
                    new_hits = max(0, candidate.consecutive_hits - 1)
                    if new_hits != candidate.consecutive_hits:
                        LOGGER.debug(
                            "Recognition weakened | label=%s hits=%d/%d similarity=%.3f",
                            candidate.label,
                            new_hits,
                            self.required_consecutive_hits,
                            result.similarity,
                        )
                    candidate.consecutive_hits = new_hits

            if (
                not candidate.confirmed
                and candidate.consecutive_hits >= self.required_consecutive_hits
            ):
                candidate.confirmed = True
                LOGGER.info(
                    "Stable recognition confirmed | label=%s similarity=%.3f",
                    candidate.label,
                    result.similarity,
                )
                confirmed_results.append(result)

            display_results.append(self._build_display_result(result, candidate))

        self._handle_unmatched_candidates(matched_indices, removed_indices)
        return StableRecognitionOutput(
            display_results=display_results,
            confirmed_results=confirmed_results,
        )

    def _drop_stale_candidates(self, now: float) -> None:
        kept_candidates: list[RecognitionCandidate] = []
        for candidate in self._candidates:
            if now - candidate.last_seen <= self.stability_time_window:
                kept_candidates.append(candidate)
            else:
                LOGGER.debug("Recognition reset due to inactivity | label=%s", candidate.label)
        self._candidates = kept_candidates

    def _handle_unmatched_candidates(
        self,
        matched_indices: set[int],
        removed_indices: set[int],
    ) -> None:
        kept_candidates: list[RecognitionCandidate] = []
        for index, candidate in enumerate(self._candidates):
            if index in removed_indices:
                continue

            if index in matched_indices:
                kept_candidates.append(candidate)
                continue

            if self._register_miss(candidate, reason="not_seen"):
                kept_candidates.append(candidate)
        self._candidates = kept_candidates

    def _register_miss(self, candidate: RecognitionCandidate, reason: str) -> bool:
        candidate.misses += 1
        if candidate.misses > self.max_misses_before_reset:
            LOGGER.debug(
                "Recognition reset due to misses | label=%s misses=%d reason=%s",
                candidate.label,
                candidate.misses,
                reason,
            )
            return False
        return True

    def _match_candidate_index(
        self,
        result: RecognitionResult,
        matched_indices: set[int],
    ) -> int | None:
        best_index: int | None = None
        best_sort_key: tuple[int, float] | None = None

        for index, candidate in enumerate(self._candidates):
            if index in matched_indices:
                continue

            distance_sq = self._bbox_center_distance_sq(result.bbox, candidate.bbox)
            if distance_sq > self.position_tolerance_sq:
                continue

            label_mismatch = 0 if candidate.person_id == result.person_id else 1
            sort_key = (label_mismatch, distance_sq)
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_index = index

        return best_index

    def _is_displayable(self, result: RecognitionResult) -> bool:
        return result.is_known and result.similarity >= self.display_min_similarity

    def _build_display_result(
        self,
        result: RecognitionResult,
        candidate: RecognitionCandidate,
    ) -> RecognitionResult:
        if candidate.confirmed:
            return replace(
                result,
                person_id=candidate.person_id,
                is_known=True,
                display_label=candidate.label,
                class_name=candidate.class_name,
                stability_state="confirmed",
            )

        if candidate.consecutive_hits > 0:
            display_label = (
                f"{candidate.label} ({candidate.consecutive_hits}/{self.required_consecutive_hits})"
            )
        else:
            display_label = f"{candidate.label} ..."

        return replace(
            result,
            person_id=candidate.person_id,
            is_known=True,
            display_label=display_label,
            class_name=candidate.class_name,
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
