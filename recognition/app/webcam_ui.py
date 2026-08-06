from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebcamOverlayInfo:
    source_text: str
    status_text: str | None
    status_color: tuple[int, int, int] | None
    focus_label: str | None
    similarity: float | None
    display_fps: float | None
    recognition_fps: float | None
    help_text: str | None
    face_count: int = 0


def initialize_webcam_window(
    window_name: str,
    *,
    start_fullscreen: bool,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Create the webcam window and apply the preferred initial size/state."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if not start_fullscreen:
            cv2.resizeWindow(window_name, int(frame_width), int(frame_height))
        if start_fullscreen:
            set_window_fullscreen(window_name, True)
        return True
    except cv2.error as exc:
        LOGGER.error("Failed to initialize webcam window '%s': %s", window_name, exc)
        return False


def set_window_fullscreen(window_name: str, enabled: bool) -> bool:
    """Toggle OpenCV fullscreen mode without raising on UI backend quirks."""
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
        return True
    except cv2.error as exc:
        LOGGER.warning(
            "Failed to set fullscreen=%s for window '%s': %s",
            enabled,
            window_name,
            exc,
        )
        return False


def draw_webcam_status_overlay(
    frame: np.ndarray,
    info: WebcamOverlayInfo,
    *,
    overlay_mode: str,
    show_source: bool,
    show_status: bool,
    show_display_fps: bool,
    show_recognition_fps: bool,
    show_help: bool,
    show_global_panel: bool,
    show_face_count: bool,
) -> np.ndarray:
    """Draw the selected webcam overlay mode with minimal per-frame overhead."""
    if overlay_mode == "minimal":
        _draw_minimal_overlay(
            frame,
            info,
            show_source=show_source,
            show_display_fps=show_display_fps,
            show_recognition_fps=show_recognition_fps,
            show_help=show_help,
            show_face_count=show_face_count,
        )
        return frame

    if not show_global_panel:
        if show_help and info.help_text:
            _draw_help_panel(frame, _truncate_text(info.help_text, 84))
        return frame

    info_lines: list[tuple[str, str, tuple[int, int, int]]] = []
    if show_status and info.status_text and info.status_color:
        info_lines.append(("Status", info.status_text, info.status_color))

    if info.focus_label:
        info_lines.append(("Match", info.focus_label, (255, 255, 255)))

    if info.similarity is not None:
        info_lines.append(("Similarity", f"{info.similarity:.3f}", (220, 220, 220)))

    if show_source and info.source_text:
        info_lines.append(("Source", _truncate_text(info.source_text, 48), (220, 220, 220)))

    if show_display_fps and info.display_fps is not None:
        info_lines.append(("Display FPS", f"{info.display_fps:.1f}", (255, 255, 0)))

    if show_recognition_fps and info.recognition_fps is not None:
        info_lines.append(("Recognition FPS", f"{info.recognition_fps:.1f}", (255, 220, 120)))

    if info_lines:
        _draw_info_panel(frame, info_lines)

    if show_help and info.help_text:
        _draw_help_panel(frame, _truncate_text(info.help_text, 84))

    return frame


def _draw_minimal_overlay(
    frame: np.ndarray,
    info: WebcamOverlayInfo,
    *,
    show_source: bool,
    show_display_fps: bool,
    show_recognition_fps: bool,
    show_help: bool,
    show_face_count: bool,
) -> None:
    parts: list[str] = []
    if show_source and info.source_text:
        parts.append(_truncate_text(info.source_text, 28))
    if show_display_fps and info.display_fps is not None:
        parts.append(f"D {info.display_fps:.1f}")
    if show_recognition_fps and info.recognition_fps is not None:
        parts.append(f"R {info.recognition_fps:.1f}")
    if show_face_count and info.face_count > 0:
        parts.append(f"Faces {info.face_count}")

    if parts:
        _draw_corner_badge(frame, " | ".join(parts))

    if show_help and info.help_text:
        _draw_help_panel(frame, _truncate_text(info.help_text, 84))


def _draw_corner_badge(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 1
    padding_x = 14
    padding_y = 10
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    panel_width = text_size[0] + padding_x * 2
    panel_height = text_size[1] + baseline + padding_y * 2
    panel_x = 18
    panel_y = 18

    _draw_translucent_panel(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.42)
    cv2.putText(
        frame,
        text,
        (panel_x + padding_x, panel_y + padding_y + text_size[1]),
        font,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )


def _draw_info_panel(
    frame: np.ndarray,
    lines: list[tuple[str, str, tuple[int, int, int]]],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.8
    text_scale = 0.58
    thickness = 1
    line_gap = 10
    padding_x = 16
    padding_y = 14
    panel_x = 18
    panel_y = 18

    label_width = 0
    value_width = 0
    line_height = 0

    for label, value, _ in lines:
        label_size, _ = cv2.getTextSize(f"{label}:", font, text_scale, thickness)
        value_size, _ = cv2.getTextSize(value, font, text_scale, thickness)
        label_width = max(label_width, label_size[0])
        value_width = max(value_width, value_size[0])
        line_height = max(line_height, label_size[1], value_size[1])

    title_size, _ = cv2.getTextSize("Live Recognition", font, title_scale, 2)
    panel_width = max(title_size[0], label_width + 14 + value_width) + (padding_x * 2)
    panel_height = (
        padding_y * 2
        + title_size[1]
        + 14
        + len(lines) * line_height
        + max(0, len(lines) - 1) * line_gap
    )

    _draw_translucent_panel(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.55)

    title_y = panel_y + padding_y + title_size[1]
    cv2.putText(
        frame,
        "Live Recognition",
        (panel_x + padding_x, title_y),
        font,
        title_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    current_y = title_y + 18
    value_x = panel_x + padding_x + label_width + 14
    for label, value, color in lines:
        current_y += line_height
        cv2.putText(
            frame,
            f"{label}:",
            (panel_x + padding_x, current_y),
            font,
            text_scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value,
            (value_x, current_y),
            font,
            text_scale,
            color,
            thickness + 1 if label == "Status" else thickness,
            cv2.LINE_AA,
        )
        current_y += line_gap


def _draw_help_panel(frame: np.ndarray, help_text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 1
    padding_x = 14
    padding_y = 10
    text_size, baseline = cv2.getTextSize(help_text, font, scale, thickness)
    panel_width = text_size[0] + padding_x * 2
    panel_height = text_size[1] + baseline + padding_y * 2
    frame_height = frame.shape[0]
    panel_x = 18
    panel_y = frame_height - panel_height - 18

    _draw_translucent_panel(frame, panel_x, panel_y, panel_width, panel_height, alpha=0.45)
    cv2.putText(
        frame,
        help_text,
        (panel_x + padding_x, panel_y + padding_y + text_size[1]),
        font,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )


def _draw_translucent_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    alpha: float,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    width = min(width, max(1, frame_width - x))
    height = min(height, max(1, frame_height - y))
    if width <= 0 or height <= 0:
        return

    overlay = frame[y : y + height, x : x + width].copy()
    overlay[:] = (18, 18, 18)
    cv2.addWeighted(
        overlay,
        alpha,
        frame[y : y + height, x : x + width],
        1.0 - alpha,
        0,
        frame[y : y + height, x : x + width],
    )


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."
