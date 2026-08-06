from __future__ import annotations

import argparse
import logging

from recognition import config
from recognition.app.database import FaceEmbeddingsDatabase
from recognition.app.face_engine import FaceEngine
from recognition.app.utils import ensure_runtime_directories, open_webcam, setup_logging
from recognition.enroll import run_enrollment
from recognition.recognize_image import run_image_recognition
from recognition.recognize_video import run_video_recognition
from recognition.recognize_webcam import run_webcam_recognition

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for local InsightFace recognition.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["enroll", "webcam", "image", "video", "self-check"],
        help="Run enrollment, webcam recognition, image recognition, video recognition, or a self-check.",
    )
    parser.add_argument("--input", help="Input path for image or video modes.")
    parser.add_argument("--output", help="Optional output path for image or video modes.")
    parser.add_argument(
        "--source", help="Optional camera index, stream URL, or video file for webcam mode."
    )
    parser.add_argument("--person", help="Optional single person folder to enroll.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Backward-compatible camera index override for webcam mode.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window for image/video modes.",
    )
    parser.add_argument(
        "--check-webcam",
        action="store_true",
        help="During self-check, also verify that the webcam can be opened.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def run_self_check(
    check_webcam: bool = False,
    camera_index: int = config.WEBCAM_INDEX,
    verbose: bool = False,
) -> int:
    setup_logging("DEBUG" if verbose else config.LOG_LEVEL)
    ensure_runtime_directories()

    database = FaceEmbeddingsDatabase(config.DATABASE_PATH)
    engine = FaceEngine()

    try:
        engine.ensure_loaded()
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("InsightFace model loaded successfully.")
    LOGGER.info("Active providers: %s", ", ".join(engine.providers))
    LOGGER.info("Expected model directory: %s", engine.expected_model_dir)
    LOGGER.info("Database path: %s", config.DATABASE_PATH)
    LOGGER.info("Enrolled identities currently stored: %d", database.count_identities())

    if check_webcam:
        capture = open_webcam(camera_index)
        try:
            if not capture.isOpened():
                LOGGER.error("Webcam self-check failed for camera index %d", camera_index)
                return 1
            LOGGER.info("Webcam opened successfully for camera index %d", camera_index)
        finally:
            capture.release()

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "enroll":
        return run_enrollment(person_name=args.person, verbose=args.verbose)

    if args.mode == "webcam":
        return run_webcam_recognition(
            source=args.source,
            camera_index=args.camera_index,
            verbose=args.verbose,
        )

    if args.mode == "image":
        if not args.input:
            parser.error("--input is required for image mode.")
        return run_image_recognition(
            input_path=args.input,
            output_path=args.output,
            display=not args.no_display,
            verbose=args.verbose,
        )

    if args.mode == "video":
        if not args.input:
            parser.error("--input is required for video mode.")
        return run_video_recognition(
            input_path=args.input,
            output_path=args.output,
            display=not args.no_display,
            verbose=args.verbose,
        )

    return run_self_check(
        check_webcam=args.check_webcam,
        camera_index=args.camera_index if args.camera_index is not None else config.WEBCAM_INDEX,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
