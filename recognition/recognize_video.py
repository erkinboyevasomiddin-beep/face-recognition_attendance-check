from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from recognition import config
from recognition.app.api_client import RecognitionEventClient
from recognition.app.database import FaceEmbeddingsDatabase
from recognition.app.face_engine import FaceEngine
from recognition.app.recognizer import FaceRecognizer, RecognitionResult
from recognition.app.stable_recognition import StableRecognitionManager
from recognition.app.utils import ensure_runtime_directories, resize_frame, setup_logging
from recognition.app.visualization import draw_recognition_results

LOGGER = logging.getLogger(__name__)


def _results_for_api_send(
    display_results: list[RecognitionResult],
    confirmed_results: list[RecognitionResult],
) -> list[RecognitionResult]:
    if config.RECOGNITION_SITE_SEND_ONLY_CONFIRMED:
        return confirmed_results
    return [result for result in display_results if result.is_known]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize faces in a video file.")
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output",
        help="Optional output video path for the annotated result.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open the OpenCV preview window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def run_video_recognition(
    input_path: str,
    output_path: str | None = None,
    display: bool = True,
    verbose: bool = False,
) -> int:
    setup_logging("DEBUG" if verbose else config.LOG_LEVEL)
    ensure_runtime_directories()

    video_path = Path(input_path)
    if not video_path.exists():
        LOGGER.error("Input video not found: %s", video_path)
        return 1

    database = FaceEmbeddingsDatabase(config.DATABASE_PATH)
    engine = FaceEngine()

    try:
        recognizer = FaceRecognizer(engine=engine, database=database)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    api_client = RecognitionEventClient()
    stable_manager = StableRecognitionManager()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        LOGGER.error("Unable to open video file: %s", video_path)
        capture.release()
        api_client.close()
        return 1

    writer: cv2.VideoWriter | None = None
    output_video_path = Path(output_path) if output_path else None
    input_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps if input_fps and input_fps > 0 else 25.0

    frame_index = 0
    last_results = []

    LOGGER.info("Processing video. Press 'q' in the video window to quit early.")

    try:
        while True:
            loop_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            frame = resize_frame(frame, config.FRAME_RESIZE_SCALE)
            should_process = config.FRAME_SKIP <= 0 or frame_index % (config.FRAME_SKIP + 1) == 0

            if should_process:
                raw_results = recognizer.recognize(frame)
                try:
                    stable_output = stable_manager.process(raw_results)
                except Exception as exc:
                    LOGGER.warning("Stable recognition processing failed: %s", exc)
                    last_results = raw_results
                    confirmed_results: list[RecognitionResult] = []
                else:
                    last_results = stable_output.display_results
                    confirmed_results = stable_output.confirmed_results

                for result in _results_for_api_send(last_results, confirmed_results):
                    api_client.send_recognition_event(
                        person_id=result.person_id,
                        display_name=result.label,
                        similarity=result.similarity,
                        apply_cooldown=True,
                        once_per_run=False,
                        class_name=result.class_name,
                        source=video_path.name,
                    )

            fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
            annotated = draw_recognition_results(frame, last_results, fps=fps)

            if output_video_path is not None:
                if writer is None:
                    output_video_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                    writer = cv2.VideoWriter(
                        str(output_video_path),
                        fourcc,
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Failed to create output video writer: {output_video_path}"
                        )

                writer.write(annotated)

            if display:
                cv2.imshow(config.WINDOW_NAME, annotated)
                pressed = cv2.waitKey(1) & 0xFF
                if pressed == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        api_client.close()

    if output_video_path is not None:
        LOGGER.info("Annotated video saved to %s", output_video_path)

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_video_recognition(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
