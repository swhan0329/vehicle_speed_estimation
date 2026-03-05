from __future__ import annotations

from pathlib import Path

import cv2 as cv


def _resolve_capture_source(video_src: str) -> int | str:
    raw = str(video_src).strip()
    path_candidate = Path(raw)

    # Keep existing files as path inputs, even if the filename is numeric.
    if path_candidate.exists():
        return raw

    if raw.lstrip("+-").isdigit():
        return int(raw)

    return raw


def open_video_source(video_src: str) -> cv.VideoCapture:
    """Open a source path or camera index with OpenCV VideoCapture."""

    source = _resolve_capture_source(video_src)
    return cv.VideoCapture(source)


def resolve_fps(capture: cv.VideoCapture, fallback_fps: float) -> float:
    fps = float(capture.get(cv.CAP_PROP_FPS))
    if fps <= 1.0:
        return fallback_fps
    return fps
