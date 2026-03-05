from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
IntPoint = tuple[int, int]
Track = list[Point]


@dataclass(frozen=True)
class LaneConfig:
    name: str
    polygon: list[IntPoint]
    px_to_meter: float
    color: tuple[int, int, int]


@dataclass(frozen=True)
class OverlayLine:
    start: IntPoint
    end: IntPoint
    color: tuple[int, int, int] = (0, 0, 255)
    thickness: int = 2


@dataclass(frozen=True)
class RuntimeConfig:
    view_polygon: list[IntPoint]
    calibration_polygon: list[IntPoint]
    lanes: list[LaneConfig]
    overlay_lines: list[OverlayLine]
    alpha: float = 0.5
    detect_interval: int = 2
    track_len: int = 2
    min_points_for_speed: int = 5
    fallback_fps: float = 30.0
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 2
    lk_criteria_count: int = 10
    lk_criteria_eps: float = 0.03
    feature_max_corners: int = 500
    feature_quality_level: float = 0.3
    feature_min_distance: int = 7
    feature_block_size: int = 7
    preview_window_name: str = "lk_track"


@dataclass(frozen=True)
class SpeedFrame:
    speeds_kmh: list[float]
    point_counts: list[int]
    avg_displacements: list[float]
