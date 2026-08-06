from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from recognition import config
from recognition.app.api_client import RecognitionEventClient
from recognition.app.database import FaceEmbeddingsDatabase
from recognition.app.face_engine import FaceEngine
from recognition.app.multi_face_tracker import MultiFaceTracker
from recognition.app.recognizer import FaceRecognizer, RecognitionResult
from recognition.app.utils import ensure_runtime_directories, resize_frame, setup_logging
from recognition.app.video_stream import ThreadedVideoStream
from recognition.app.visualization import draw_recognition_results
from recognition.app.webcam_event_worker import AsyncConfirmedEventDispatcher
from recognition.app.webcam_ui import (
    WebcamOverlayInfo,
    draw_webcam_status_overlay,
    initialize_webcam_window,
    set_window_fullscreen,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionJobResult:
    results: list[RecognitionResult]
    recognition_fps: float
    frame_id: int
    source_frame: np.ndarray
    generation: int


@dataclass(frozen=True)
class CachedRecognitionResults:
    results: list[RecognitionResult]
    timestamp: float
    recognition_fps: float
    active_track_count: int


def _configure_webcam_logging(verbose: bool) -> bool:
    quiet_mode = bool(config.QUIET_MODE and not verbose)
    if quiet_mode:
        logging.getLogger("recognition.app.api_client").setLevel(logging.WARNING)
        logging.getLogger("recognition.app.video_source").setLevel(logging.WARNING)
        logging.getLogger("recognition.app.webcam_event_worker").setLevel(logging.WARNING)
    return quiet_mode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces from a camera, stream URL, or video file."
    )
    parser.add_argument(
        "--source",
        help=(
            "Video source override. Examples: 0, 1, rtsp://192.0.2.10:554/stream, "
            "http://host/stream, test.mp4"
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Backward-compatible camera index override for local camera devices.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _scale_recognition_results(
    results: list[RecognitionResult],
    scale_x: float,
    scale_y: float,
) -> list[RecognitionResult]:
    if scale_x == 1.0 and scale_y == 1.0:
        return results

    scaled_results: list[RecognitionResult] = []
    for result in results:
        x1, y1, x2, y2 = result.bbox
        scaled_results.append(
            replace(
                result,
                bbox=(
                    int(round(x1 * scale_x)),
                    int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)),
                    int(round(y2 * scale_y)),
                ),
            )
        )
    return scaled_results


def _prepare_recognition_frame(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    recognition_scale = float(config.WEBCAM_RECOGNITION_SCALE)
    if recognition_scale <= 0:
        LOGGER.warning(
            "Invalid WEBCAM_RECOGNITION_SCALE %.3f. Falling back to 1.0.", recognition_scale
        )
        recognition_scale = 1.0

    if recognition_scale == 1.0:
        return frame, 1.0, 1.0

    recognition_frame = resize_frame(frame, recognition_scale)
    scale_x = frame.shape[1] / recognition_frame.shape[1]
    scale_y = frame.shape[0] / recognition_frame.shape[0]
    return recognition_frame, scale_x, scale_y


def _run_recognition_job(
    recognizer: FaceRecognizer,
    frame: np.ndarray,
    frame_id: int,
    generation: int,
) -> RecognitionJobResult:
    recognition_frame, scale_x, scale_y = _prepare_recognition_frame(frame)
    started_at = time.perf_counter()
    results = recognizer.recognize(recognition_frame)
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    return RecognitionJobResult(
        results=_scale_recognition_results(results, scale_x, scale_y),
        recognition_fps=1.0 / elapsed,
        frame_id=frame_id,
        source_frame=frame,
        generation=generation,
    )


def _should_display_cached_results(cache: CachedRecognitionResults | None) -> bool:
    if cache is None:
        return False
    age = time.perf_counter() - cache.timestamp
    return age <= config.WEBCAM_RESULT_TTL_SECONDS


def _resolve_runtime_video_source(
    source: str | int | None,
    camera_index: int | None,
) -> str | int:
    if source is not None:
        return source
    if camera_index is not None:
        return camera_index
    return config.VIDEO_SOURCE


def _resolve_event_source_name(source_text: str | None, configured_name: str) -> str | None:
    configured_name = configured_name.strip()
    if configured_name:
        return configured_name
    if source_text is None:
        return None
    normalized = source_text.strip()
    return normalized or None


def _results_for_api_send(
    display_results: list[RecognitionResult],
    confirmed_results: list[RecognitionResult],
) -> list[RecognitionResult]:
    if config.RECOGNITION_SITE_SEND_ONLY_CONFIRMED:
        return confirmed_results
    return [result for result in display_results if result.is_known]


def _resolve_status_overlay_info(
    results: list[RecognitionResult],
    *,
    source_text: str,
    display_fps: float | None,
    recognition_fps: float | None,
) -> WebcamOverlayInfo:
    confirmed_results = [result for result in results if result.stability_state == "confirmed"]
    tentative_results = [result for result in results if result.stability_state == "tentative"]

    focus_result: RecognitionResult | None = None
    status_text = "Scanning"
    status_color = (210, 210, 210)

    if confirmed_results:
        focus_result = max(confirmed_results, key=lambda result: result.similarity)
        status_text = "Confirmed"
        status_color = (90, 220, 120)
    elif tentative_results:
        focus_result = max(tentative_results, key=lambda result: result.similarity)
        status_text = "Verifying"
        status_color = (0, 185, 255)

    return WebcamOverlayInfo(
        source_text=source_text,
        status_text=status_text,
        status_color=status_color,
        focus_label=focus_result.display_label if focus_result is not None else None,
        similarity=focus_result.similarity if focus_result is not None else None,
        display_fps=display_fps,
        recognition_fps=recognition_fps,
        help_text=_build_help_text(),
        face_count=len(results),
    )


def _build_help_text() -> str:
    if not config.WEBCAM_SHOW_HELP_OVERLAY:
        return ""
    if config.WEBCAM_ENABLE_RESET_HOTKEY:
        return "F = fullscreen | R = reset | Q / Esc = quit"
    return "F = fullscreen | Q / Esc = quit"


def run_webcam_recognition(
    source: str | int | None = None,
    camera_index: int | None = None,
    verbose: bool = False,
) -> int:
    setup_logging("DEBUG" if verbose else config.LOG_LEVEL)
    quiet_mode = _configure_webcam_logging(verbose)
    ensure_runtime_directories()

    database = FaceEmbeddingsDatabase(config.DATABASE_PATH)
    engine = FaceEngine(detection_size=config.WEBCAM_DET_SIZE)
    overlay_mode = config.WEBCAM_OVERLAY_MODE.strip().casefold()
    if overlay_mode not in {"minimal", "full"}:
        LOGGER.warning(
            "Unsupported WEBCAM_OVERLAY_MODE '%s'. Falling back to 'minimal'.",
            config.WEBCAM_OVERLAY_MODE,
        )
        overlay_mode = "minimal"

    try:
        recognizer = FaceRecognizer(
            engine=engine,
            database=database,
            smoothing_enabled=False,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    api_client = RecognitionEventClient()
    face_tracker = MultiFaceTracker()
    event_dispatcher = AsyncConfirmedEventDispatcher(api_client=api_client)
    selected_source = _resolve_runtime_video_source(source=source, camera_index=camera_index)

    stream = ThreadedVideoStream(
        source=selected_source,
        frame_width=config.WEBCAM_FRAME_WIDTH,
        frame_height=config.WEBCAM_FRAME_HEIGHT,
        source_type=config.VIDEO_SOURCE_TYPE,
        threaded=config.WEBCAM_ENABLE_THREADED_CAPTURE,
        auto_reconnect=config.CAMERA_AUTO_RECONNECT,
        reconnect_interval_seconds=config.CAMERA_RECONNECT_INTERVAL_SECONDS,
        max_read_failures_before_reconnect=config.CAMERA_MAX_READ_FAILURES_BEFORE_RECONNECT,
        read_failure_retry_delay_seconds=config.CAMERA_READ_FAILURE_RETRY_DELAY_SECONDS,
        quiet_mode=quiet_mode,
    )
    if not stream.start():
        event_dispatcher.close()
        return 1

    if not initialize_webcam_window(
        config.WINDOW_NAME,
        start_fullscreen=config.WEBCAM_START_FULLSCREEN,
        frame_width=config.WEBCAM_FRAME_WIDTH,
        frame_height=config.WEBCAM_FRAME_HEIGHT,
    ):
        event_dispatcher.close()
        stream.stop()
        cv2.destroyAllWindows()
        return 1

    resolved_source = stream.resolved_source
    source_text = (
        resolved_source.description if resolved_source is not None else str(selected_source)
    )
    event_source_name = _resolve_event_source_name(
        resolved_source.event_source if resolved_source is not None else None,
        config.VIDEO_SOURCE_NAME,
    )
    if resolved_source is not None:
        if quiet_mode:
            LOGGER.info("Starting kiosk webcam mode | source=%s", resolved_source.capture_value)
        else:
            LOGGER.info(
                "Starting live recognition | source_type=%s | source=%s",
                resolved_source.source_type,
                resolved_source.capture_value,
            )
    else:
        LOGGER.info("Starting live recognition. Press 'q' in the video window to quit.")

    if not quiet_mode:
        LOGGER.info("Press 'q' in the video window to quit.")
        LOGGER.info(
            "Webcam performance config | threaded_capture=%s process_every=%d recognition_scale=%.2f det_size=%s result_ttl=%.2fs event_queue=%d",
            config.WEBCAM_ENABLE_THREADED_CAPTURE,
            config.WEBCAM_PROCESS_EVERY_N_FRAMES,
            config.WEBCAM_RECOGNITION_SCALE,
            config.WEBCAM_DET_SIZE,
            config.WEBCAM_RESULT_TTL_SECONDS,
            config.WEBCAM_MAX_QUEUE_SIZE,
        )

    process_every_n_frames = max(1, int(config.WEBCAM_PROCESS_EVERY_N_FRAMES))
    recognition_future: Future[RecognitionJobResult] | None = None
    cache: CachedRecognitionResults | None = None
    display_frame_index = 0
    display_fps = 0.0
    latest_recognition_fps: float | None = None
    display_counter = 0
    fps_window_started_at = time.perf_counter()
    fps_last_logged_at = fps_window_started_at
    last_processed_frame_id = -1
    last_displayed_frame_id = -1
    reset_generation = 0
    fullscreen_enabled = config.WEBCAM_START_FULLSCREEN
    stop_message: str | None = None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="webcam-recognition") as executor:
        try:
            try:
                while True:
                    if recognition_future is not None and recognition_future.done():
                        try:
                            job_result = recognition_future.result()
                        except Exception as exc:
                            LOGGER.warning("Webcam recognition worker failed: %s", exc)
                        else:
                            if job_result.generation != reset_generation:
                                LOGGER.debug(
                                    "Discarding stale recognition result from generation %d; current generation is %d.",
                                    job_result.generation,
                                    reset_generation,
                                )
                                recognition_future = None
                                continue
                            try:
                                track_output = face_tracker.update(job_result.results)
                            except Exception as exc:
                                LOGGER.warning("Multi-face tracking failed: %s", exc)
                                display_results = job_result.results
                                confirmed_results: list[RecognitionResult] = []
                                active_track_count = len(job_result.results)
                            else:
                                display_results = track_output.display_results
                                confirmed_results = track_output.confirmed_results
                                active_track_count = track_output.active_track_count

                            cache = CachedRecognitionResults(
                                results=display_results,
                                timestamp=time.perf_counter(),
                                recognition_fps=job_result.recognition_fps,
                                active_track_count=active_track_count,
                            )
                            latest_recognition_fps = job_result.recognition_fps
                            last_processed_frame_id = job_result.frame_id
                            LOGGER.debug(
                                "Recognition pass | frame_id=%d visible_faces=%d active_tracks=%d recognition_fps=%.2f",
                                job_result.frame_id,
                                len(display_results),
                                active_track_count,
                                job_result.recognition_fps,
                            )
                            for result in _results_for_api_send(display_results, confirmed_results):
                                event_dispatcher.submit_confirmed_event(
                                    person_id=result.person_id,
                                    display_name=result.label,
                                    class_name=result.class_name,
                                    similarity=result.similarity,
                                    frame=job_result.source_frame,
                                    bbox=result.bbox,
                                    source=event_source_name,
                                )
                        finally:
                            recognition_future = None

                    stream_frame = stream.read()
                    if stream_frame is None:
                        if stream.ended and recognition_future is None:
                            LOGGER.info("Video source ended. Exiting live recognition loop.")
                            break
                        time.sleep(0.01)
                        continue

                    if stream_frame.frame_id == last_displayed_frame_id:
                        if stream.ended and recognition_future is None:
                            LOGGER.info("Video source ended. Exiting live recognition loop.")
                            break
                        time.sleep(0.001)
                        continue
                    last_displayed_frame_id = stream_frame.frame_id

                    frame = resize_frame(stream_frame.frame, config.FRAME_RESIZE_SCALE)

                    should_submit_recognition = (
                        recognition_future is None
                        and display_frame_index % process_every_n_frames == 0
                        and stream_frame.frame_id != last_processed_frame_id
                    )
                    if should_submit_recognition:
                        recognition_input = frame.copy()
                        recognition_future = executor.submit(
                            _run_recognition_job,
                            recognizer,
                            recognition_input,
                            stream_frame.frame_id,
                            reset_generation,
                        )

                    results_to_display = (
                        cache.results
                        if cache is not None and _should_display_cached_results(cache)
                        else []
                    )
                    active_track_count = (
                        cache.active_track_count
                        if _should_display_cached_results(cache) and cache is not None
                        else 0
                    )

                    display_counter += 1
                    now = time.perf_counter()
                    fps_elapsed = now - fps_window_started_at
                    if fps_elapsed >= 0.5:
                        display_fps = display_counter / fps_elapsed
                        LOGGER.debug("Webcam display FPS: %.2f", display_fps)
                        display_counter = 0
                        fps_window_started_at = now
                        if not quiet_mode and now - fps_last_logged_at >= 5.0:
                            LOGGER.info(
                                "Webcam performance | display_fps=%.2f recognition_fps=%s event_queue=%d",
                                display_fps,
                                f"{latest_recognition_fps:.2f}"
                                if latest_recognition_fps is not None
                                else "--",
                                event_dispatcher.queue_size,
                            )
                            fps_last_logged_at = now

                    annotated = draw_recognition_results(
                        frame,
                        results_to_display,
                        fps=None,
                        recognition_fps=None,
                        copy_frame=False,
                        show_similarity=overlay_mode == "full",
                        show_face_labels=config.WEBCAM_SHOW_FACE_LABELS,
                        use_display_label=config.WEBCAM_SHOW_PER_FACE_STATE
                        or overlay_mode == "full",
                    )
                    overlay_info = _resolve_status_overlay_info(
                        results_to_display,
                        source_text=source_text,
                        display_fps=display_fps
                        if config.WEBCAM_SHOW_FPS and config.WEBCAM_SHOW_DISPLAY_FPS
                        else None,
                        recognition_fps=latest_recognition_fps
                        if config.WEBCAM_SHOW_FPS and config.WEBCAM_SHOW_RECOGNITION_FPS
                        else None,
                    )
                    draw_webcam_status_overlay(
                        annotated,
                        overlay_info,
                        overlay_mode=overlay_mode,
                        show_source=config.WEBCAM_SHOW_SOURCE,
                        show_status=config.WEBCAM_SHOW_STATUS,
                        show_display_fps=config.WEBCAM_SHOW_FPS and config.WEBCAM_SHOW_DISPLAY_FPS,
                        show_recognition_fps=config.WEBCAM_SHOW_FPS
                        and config.WEBCAM_SHOW_RECOGNITION_FPS,
                        show_help=config.WEBCAM_SHOW_HELP_OVERLAY,
                        show_global_panel=config.WEBCAM_SHOW_GLOBAL_PANEL,
                        show_face_count=config.WEBCAM_SHOW_FACE_COUNT,
                    )

                    cv2.imshow(config.WINDOW_NAME, annotated)
                    pressed = cv2.waitKey(1) & 0xFF
                    if pressed in (ord("q"), ord("Q"), 27):
                        stop_message = "Stopped by user."
                        break
                    if pressed in (ord("f"), ord("F")):
                        requested_fullscreen = not fullscreen_enabled
                        if set_window_fullscreen(config.WINDOW_NAME, requested_fullscreen):
                            fullscreen_enabled = requested_fullscreen
                            LOGGER.info(
                                "Fullscreen %s.",
                                "enabled" if fullscreen_enabled else "disabled",
                            )
                    if pressed in (ord("r"), ord("R")) and config.WEBCAM_ENABLE_RESET_HOTKEY:
                        reset_generation += 1
                        face_tracker.reset()
                        recognizer.reset_state()
                        cache = None
                        LOGGER.info("Per-face recognition state reset by user.")

                    display_frame_index += 1
            except KeyboardInterrupt:
                stop_message = "Stopped by user."
        finally:
            if recognition_future is not None:
                recognition_future.cancel()
            event_dispatcher.close()
            stream.stop()
            cv2.destroyAllWindows()

    if stop_message:
        LOGGER.info(stop_message)

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_webcam_recognition(
        source=args.source,
        camera_index=args.camera_index,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
