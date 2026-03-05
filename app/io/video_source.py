from __future__ import annotations

import cv2 as cv

import video


def open_video_source(video_src: str) -> cv.VideoCapture:
    """Open a source path or camera index using the existing helper."""

    return video.create_capture(video_src)


def resolve_fps(capture: cv.VideoCapture, fallback_fps: float) -> float:
    fps = float(capture.get(cv.CAP_PROP_FPS))
    if fps <= 1.0:
        return fallback_fps
    return fps
