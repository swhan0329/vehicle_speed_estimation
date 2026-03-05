from __future__ import annotations

from typing import Sequence


class SpeedSmoother:
    """Smooth lane speeds while preserving previous values when evidence is low."""

    def __init__(self, lane_count: int, smooth_coef: float = 0.5) -> None:
        self._values = [0.0] * lane_count
        self._initialized = [False] * lane_count
        self._smooth_coef = float(smooth_coef)

    def values(self) -> list[float]:
        return list(self._values)

    def update(
        self,
        raw_speeds: Sequence[float],
        point_counts: Sequence[int],
        *,
        update_enabled: bool,
        min_points_for_speed: int,
    ) -> list[float]:
        if not update_enabled:
            return self.values()

        for index, raw_speed in enumerate(raw_speeds):
            if point_counts[index] <= min_points_for_speed:
                continue
            if not self._initialized[index]:
                self._values[index] = float(raw_speed)
                self._initialized[index] = True
            else:
                self._values[index] = (
                    self._smooth_coef * self._values[index]
                    + (1.0 - self._smooth_coef) * float(raw_speed)
                )

        return self.values()
