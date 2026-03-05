from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np

from app.types import LaneConfig, SpeedFrame, Track


class LaneSpeedEstimator:
    """Estimate lane-wise speed from short point tracks."""

    def __init__(self, lanes: Sequence[LaneConfig], fps: float) -> None:
        self._lanes = list(lanes)
        self._fps = float(fps)
        self._lane_polygons = [np.asarray(lane.polygon, dtype=np.int32) for lane in self._lanes]

    def _find_lane_index(self, point: tuple[float, float]) -> int | None:
        for index, polygon in enumerate(self._lane_polygons):
            if cv.pointPolygonTest(polygon, point, False) > 0:
                return index
        return None

    def estimate(self, tracks: Sequence[Track]) -> SpeedFrame:
        lane_count = len(self._lanes)
        point_counts = [0] * lane_count
        lane_total_displacements = [0.0] * lane_count

        for track in tracks:
            if len(track) < 2:
                continue

            displacement = float(np.linalg.norm(np.subtract(track[-1], track[-2])))
            lane_index = self._find_lane_index(track[-1])
            if lane_index is None:
                continue

            point_counts[lane_index] += 1
            lane_total_displacements[lane_index] += displacement

        avg_displacements = [0.0] * lane_count
        speeds_kmh = [0.0] * lane_count

        for index, lane in enumerate(self._lanes):
            if point_counts[index] == 0:
                continue
            avg_disp = lane_total_displacements[index] / point_counts[index]
            avg_displacements[index] = avg_disp
            speeds_kmh[index] = avg_disp * lane.px_to_meter * self._fps * 3.6

        return SpeedFrame(
            speeds_kmh=speeds_kmh,
            point_counts=point_counts,
            avg_displacements=avg_displacements,
        )
