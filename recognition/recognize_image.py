from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from recognition import config
from recognition.app.api_client import RecognitionEventClient
from recognition.app.database import FaceEmbeddingsDatabase
from recognition.app.face_engine import FaceEngine
from recognition.app.recognizer import FaceRecognizer
from recognition.app.utils import ensure_runtime_directories, read_image, setup_logging, write_image
from recognition.app.visualization import draw_recognition_results

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize faces in a single image.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        help="Optional path to save an annotated output image.",
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


def run_image_recognition(
    input_path: str,
    output_path: str | None = None,
    display: bool = True,
    verbose: bool = False,
) -> int:
    setup_logging("DEBUG" if verbose else config.LOG_LEVEL)
    ensure_runtime_directories()

    image_path = Path(input_path)
    if not image_path.exists():
        LOGGER.error("Input image not found: %s", image_path)
        return 1

    image = read_image(image_path)
    if image is None:
        LOGGER.error("Failed to load image: %s", image_path)
        return 1

    database = FaceEmbeddingsDatabase(config.DATABASE_PATH)
    engine = FaceEngine()

    try:
        recognizer = FaceRecognizer(engine=engine, database=database)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    api_client = RecognitionEventClient()

    try:
        results = recognizer.recognize(image)
        annotated = draw_recognition_results(image, results)

        if results:
            for result in results:
                LOGGER.info(
                    "Face detected | label=%s | similarity=%.4f | bbox=%s",
                    result.label,
                    result.similarity,
                    result.bbox,
                )
                api_client.send_recognition_event(
                    person_id=result.person_id,
                    display_name=result.label,
                    similarity=result.similarity,
                    apply_cooldown=False,
                    once_per_run=True,
                    class_name=result.class_name,
                    source=image_path.name,
                )
        else:
            LOGGER.info("No faces were detected in %s", image_path)

        if output_path:
            destination = Path(output_path)
            write_image(destination, annotated)
            LOGGER.info("Annotated image saved to %s", destination)

        if display:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(config.WINDOW_NAME, annotated)
            LOGGER.info("Press any key in the image window to close it.")
            cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
        api_client.close()
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_image_recognition(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
