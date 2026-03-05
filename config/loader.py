from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.types import LaneConfig, OverlayLine, RuntimeConfig


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc

    parsed = yaml.safe_load(content)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return parsed


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, override_value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], override_value)
            else:
                merged[key] = override_value
        return merged
    return override


def _as_point(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ConfigError(f"{field_name} must be [x, y]")
    x, y = raw
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ConfigError(f"{field_name} coordinates must be numeric")
    return int(round(x)), int(round(y))


def _as_polygon(raw: Any, field_name: str) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        raise ConfigError(f"{field_name} must be a list of points")
    polygon = [_as_point(point, f"{field_name}[]") for point in raw]
    if len(polygon) < 3:
        raise ConfigError(f"{field_name} must contain at least 3 points")
    return polygon


def _as_color(raw: Any, field_name: str) -> tuple[int, int, int]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ConfigError(f"{field_name} must be [b, g, r]")
    channels: list[int] = []
    for value in raw:
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{field_name} channels must be numeric")
        clamped = max(0, min(255, int(round(value))))
        channels.append(clamped)
    return channels[0], channels[1], channels[2]


def _as_positive_int(value: Any, field_name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field_name} must be an integer >= {minimum}")
    return value


def _as_positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be numeric")
    casted = float(value)
    if casted <= 0:
        raise ConfigError(f"{field_name} must be > 0")
    return casted


def _parse_config(raw: dict[str, Any]) -> RuntimeConfig:
    runtime = raw.get("runtime", {})
    lk = raw.get("lk", {})
    feature = raw.get("feature", {})
    polygons = raw.get("polygons", {})
    lanes_raw = raw.get("lanes", [])
    overlay = raw.get("overlay", {})

    if not isinstance(runtime, dict) or not isinstance(lk, dict):
        raise ConfigError("runtime and lk sections must be mappings")
    if not isinstance(feature, dict) or not isinstance(polygons, dict):
        raise ConfigError("feature and polygons sections must be mappings")
    if not isinstance(overlay, dict):
        raise ConfigError("overlay section must be a mapping")

    view_polygon = _as_polygon(polygons.get("view", []), "polygons.view")
    calibration_polygon = _as_polygon(
        polygons.get("calibration", []), "polygons.calibration"
    )

    if not isinstance(lanes_raw, list) or len(lanes_raw) == 0:
        raise ConfigError("lanes must be a non-empty list")

    lanes: list[LaneConfig] = []
    for index, lane_raw in enumerate(lanes_raw):
        if not isinstance(lane_raw, dict):
            raise ConfigError(f"lanes[{index}] must be a mapping")

        name = lane_raw.get("name", f"lane-{index + 1}")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"lanes[{index}].name must be a non-empty string")

        polygon = _as_polygon(lane_raw.get("polygon", []), f"lanes[{index}].polygon")
        px_to_meter = _as_positive_float(
            lane_raw.get("px_to_meter"), f"lanes[{index}].px_to_meter"
        )
        color = _as_color(lane_raw.get("color", [255, 255, 255]), f"lanes[{index}].color")

        lanes.append(
            LaneConfig(
                name=name,
                polygon=polygon,
                px_to_meter=px_to_meter,
                color=color,
            )
        )

    lines_raw = overlay.get("lines", [])
    if not isinstance(lines_raw, list):
        raise ConfigError("overlay.lines must be a list")

    overlay_lines: list[OverlayLine] = []
    for index, line_raw in enumerate(lines_raw):
        if not isinstance(line_raw, dict):
            raise ConfigError(f"overlay.lines[{index}] must be a mapping")
        start = _as_point(line_raw.get("start"), f"overlay.lines[{index}].start")
        end = _as_point(line_raw.get("end"), f"overlay.lines[{index}].end")
        color = _as_color(
            line_raw.get("color", [0, 0, 255]), f"overlay.lines[{index}].color"
        )
        thickness = _as_positive_int(
            line_raw.get("thickness", 2), f"overlay.lines[{index}].thickness"
        )
        overlay_lines.append(
            OverlayLine(start=start, end=end, color=color, thickness=thickness)
        )

    return RuntimeConfig(
        view_polygon=view_polygon,
        calibration_polygon=calibration_polygon,
        lanes=lanes,
        overlay_lines=overlay_lines,
        alpha=float(runtime.get("alpha", 0.5)),
        detect_interval=_as_positive_int(runtime.get("detect_interval", 2), "runtime.detect_interval"),
        track_len=_as_positive_int(runtime.get("track_len", 2), "runtime.track_len", minimum=2),
        min_points_for_speed=_as_positive_int(
            runtime.get("min_points_for_speed", 5), "runtime.min_points_for_speed"
        ),
        fallback_fps=_as_positive_float(runtime.get("fallback_fps", 30.0), "runtime.fallback_fps"),
        lk_win_size=tuple(
            _as_point(lk.get("win_size", [15, 15]), "lk.win_size")
        ),
        lk_max_level=_as_positive_int(lk.get("max_level", 2), "lk.max_level"),
        lk_criteria_count=_as_positive_int(
            lk.get("criteria_count", 10), "lk.criteria_count"
        ),
        lk_criteria_eps=_as_positive_float(lk.get("criteria_eps", 0.03), "lk.criteria_eps"),
        feature_max_corners=_as_positive_int(
            feature.get("max_corners", 500), "feature.max_corners"
        ),
        feature_quality_level=_as_positive_float(
            feature.get("quality_level", 0.3), "feature.quality_level"
        ),
        feature_min_distance=_as_positive_int(
            feature.get("min_distance", 7), "feature.min_distance"
        ),
        feature_block_size=_as_positive_int(
            feature.get("block_size", 7), "feature.block_size"
        ),
        preview_window_name=str(runtime.get("preview_window_name", "lk_track")),
    )


def load_config(
    config_path: Path | str | None = None,
    default_path: Path | str | None = None,
) -> RuntimeConfig:
    base_path = Path(default_path) if default_path else DEFAULT_CONFIG_PATH
    base_config = _read_yaml(base_path)

    if config_path is None:
        merged = base_config
    else:
        override = _read_yaml(Path(config_path))
        merged = _deep_merge(base_config, override)

    return _parse_config(merged)
