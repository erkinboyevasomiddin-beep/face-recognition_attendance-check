from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from recognition.app.recognizer import RecognitionResult


def _draw_label_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = origin
    label_top = y - text_height - baseline - 6
    label_bottom = y

    if label_top < 0:
        label_top = y
        label_bottom = y + text_height + baseline + 6

    top_left = (x, label_top)
    bottom_right = (x + text_width + 8, label_bottom)
    cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x + 4, label_bottom - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_recognition_results(
    frame: np.ndarray,
    results: Sequence[RecognitionResult],
    fps: float | None = None,
    recognition_fps: float | None = None,
    copy_frame: bool = True,
    show_similarity: bool = True,
    show_face_labels: bool = True,
    use_display_label: bool = True,
) -> np.ndarray:
    """Draw face boxes, labels, and optional performance overlays."""
    annotated = frame.copy() if copy_frame else frame

    for result in results:
        x1, y1, x2, y2 = result.bbox
        if result.stability_state == "confirmed":
            color = (0, 180, 0)
        elif result.stability_state == "tentative":
            color = (0, 165, 255)
        elif result.stability_state == "unknown":
            color = (128, 128, 128)
        else:
            color = (0, 180, 0) if result.is_known else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if show_face_labels:
            label_text = (
                result.display_label if use_display_label and result.display_label else result.label
            )
            label = label_text
            if show_similarity:
                label = f"{label_text} | sim={result.similarity:.3f}"
            _draw_label_box(annotated, label, (x1, y1), color)

    if fps is not None:
        cv2.putText(
            annotated,
            f"Display FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if recognition_fps is not None:
        cv2.putText(
            annotated,
            f"Recognition FPS: {recognition_fps:.1f}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 120),
            2,
            cv2.LINE_AA,
        )

    return annotated
