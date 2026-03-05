from __future__ import annotations

from typing import Sequence

import cv2 as cv
import numpy as np

from app.types import Track


def detect_features(
    frame_gray: np.ndarray,
    tracks: Sequence[Track],
    *,
    max_corners: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
) -> list[tuple[float, float]]:
    """Detect new track seed points while masking currently tracked points."""

    mask = np.full_like(frame_gray, 255)
    for track in tracks:
        if not track:
            continue
        x, y = np.int32(track[-1])
        cv.circle(mask, (int(x), int(y)), 3, 0, -1)

    points = cv.goodFeaturesToTrack(
        frame_gray,
        mask=mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )
    if points is None:
        return []

    return [(float(x), float(y)) for x, y in np.float32(points).reshape(-1, 2)]
