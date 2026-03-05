from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np

from app.detect.gftt_detector import detect_features
from app.types import Track


class LKTracker:
    """Maintain Lucas-Kanade point tracks across frames."""

    def __init__(
        self,
        *,
        track_len: int,
        detect_interval: int,
        lk_params: dict[str, Any],
        feature_params: dict[str, Any],
    ) -> None:
        self._track_len = track_len
        self._detect_interval = detect_interval
        self._lk_params = lk_params
        self._feature_params = feature_params
        self._tracks: list[Track] = []
        self._prev_gray: np.ndarray | None = None

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def process_frame(self, frame_gray: np.ndarray, frame_idx: int) -> list[Track]:
        if self._tracks and self._prev_gray is not None:
            p0 = np.float32([track[-1] for track in self._tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv.calcOpticalFlowPyrLK(
                self._prev_gray,
                frame_gray,
                p0,
                None,
                **self._lk_params,
            )

            if p1 is not None:
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(
                    frame_gray,
                    self._prev_gray,
                    p1,
                    None,
                    **self._lk_params,
                )
            else:
                p0r = None

            if p1 is not None and p0r is not None:
                distance = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = distance < 1
                new_tracks: list[Track] = []
                for track, (x, y), is_good in zip(
                    self._tracks,
                    p1.reshape(-1, 2),
                    good,
                ):
                    if not is_good:
                        continue
                    track.append((float(x), float(y)))
                    if len(track) > self._track_len:
                        del track[0]
                    new_tracks.append(track)
                self._tracks = new_tracks
            else:
                self._tracks = []

        if frame_idx % self._detect_interval == 0:
            new_points = detect_features(
                frame_gray,
                self._tracks,
                max_corners=int(self._feature_params["maxCorners"]),
                quality_level=float(self._feature_params["qualityLevel"]),
                min_distance=int(self._feature_params["minDistance"]),
                block_size=int(self._feature_params["blockSize"]),
            )
            for x, y in new_points:
                self._tracks.append([(x, y)])

        self._prev_gray = frame_gray
        return self._tracks
