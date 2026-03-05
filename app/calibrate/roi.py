from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from app.io.video_source import open_video_source
from config.loader import DEFAULT_CONFIG_PATH

DEFAULT_LANE_COUNT = 5


def build_calibration_labels(lane_count: int) -> list[str]:
    if lane_count < 1:
        raise ValueError("lane_count must be >= 1")
    return ["view", "calibration"] + [f"lane-{index}" for index in range(1, lane_count + 1)]


def _hsv_to_bgr(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(blue * 255), int(green * 255), int(red * 255)


def build_step_colors(labels: list[str]) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    for label in labels:
        if label == "view":
            colors[label] = (255, 180, 0)
            continue
        if label == "calibration":
            colors[label] = (0, 215, 255)
            continue

        lane_index = int(label.split("-")[1]) - 1
        hue = (lane_index * 0.137) % 1.0
        colors[label] = _hsv_to_bgr(hue=hue, saturation=0.8, value=1.0)
    return colors


def map_display_point_to_frame(
    point: tuple[int, int],
    display_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int]:
    disp_x, disp_y = point
    disp_w, disp_h = display_size
    frame_w, frame_h = frame_size

    if disp_w <= 1 or disp_h <= 1 or frame_w <= 1 or frame_h <= 1:
        return max(0, disp_x), max(0, disp_y)

    mapped_x = int(round(disp_x * (frame_w - 1) / (disp_w - 1)))
    mapped_y = int(round(disp_y * (frame_h - 1) / (disp_h - 1)))
    mapped_x = max(0, min(frame_w - 1, mapped_x))
    mapped_y = max(0, min(frame_h - 1, mapped_y))
    return mapped_x, mapped_y


def _dim_color(color: tuple[int, int, int], ratio: float = 0.55) -> tuple[int, int, int]:
    return (
        int(color[0] * ratio),
        int(color[1] * ratio),
        int(color[2] * ratio),
    )


class PolygonCollector:
    def __init__(
        self,
        frame: np.ndarray,
        labels: list[str],
        step_colors: dict[str, tuple[int, int, int]],
    ) -> None:
        self._base_frame = frame
        self._labels = labels
        self._step_colors = step_colors
        self._current_index = 0
        self._polygons: dict[str, list[tuple[int, int]]] = {label: [] for label in labels}
        self._window = "roi_calibration"
        cv.namedWindow(self._window, cv.WINDOW_NORMAL)
        frame_h, frame_w = self._base_frame.shape[:2]
        cv.resizeWindow(self._window, frame_w, frame_h)
        cv.setMouseCallback(self._window, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv.EVENT_LBUTTONDOWN and self._current_index < len(self._labels):
            mapped_x, mapped_y = x, y
            try:
                _wx, _wy, disp_w, disp_h = cv.getWindowImageRect(self._window)
                frame_h, frame_w = self._base_frame.shape[:2]
                mapped_x, mapped_y = map_display_point_to_frame(
                    point=(x, y),
                    display_size=(disp_w, disp_h),
                    frame_size=(frame_w, frame_h),
                )
            except cv.error:
                mapped_x, mapped_y = x, y

            label = self._labels[self._current_index]
            self._polygons[label].append((mapped_x, mapped_y))

    def _draw(self) -> np.ndarray:
        canvas = self._base_frame.copy()

        for index, label in enumerate(self._labels):
            points = self._polygons[label]
            if not points:
                continue

            base_color = self._step_colors[label]
            color = base_color if index == self._current_index else _dim_color(base_color)
            polyline = np.asarray(points, dtype=np.int32)
            cv.polylines(canvas, [polyline], False, color, 2)

            if len(points) >= 3:
                cv.polylines(canvas, [polyline], True, color, 2)

            for point in points:
                cv.circle(canvas, point, 4, color, -1)

        if self._current_index < len(self._labels):
            current_label = self._labels[self._current_index]
            current_color = self._step_colors[current_label]
            msg1 = (
                f"Step {self._current_index + 1}/{len(self._labels)}: "
                f"{current_label} (clockwise recommended)"
            )
            msg2 = "Click add | n/Enter next | u/z undo | r reset | s save | q/ESC quit"
            cv.putText(canvas, msg1, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv.LINE_AA)
            cv.putText(canvas, msg2, (20, 58), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
        else:
            cv.putText(
                canvas,
                "All polygons captured. Press s to save or q to quit.",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        return canvas

    def run(self) -> dict[str, list[tuple[int, int]]] | None:
        while True:
            cv.imshow(self._window, self._draw())
            key = cv.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                return None

            if key in (ord("u"), ord("z"), 26) and self._current_index < len(self._labels):
                label = self._labels[self._current_index]
                if self._polygons[label]:
                    self._polygons[label].pop()

            if key in (ord("r"), ord("R")) and self._current_index < len(self._labels):
                label = self._labels[self._current_index]
                self._polygons[label] = []

            if key in (ord("n"), ord("N"), 13) and self._current_index < len(self._labels):
                label = self._labels[self._current_index]
                if len(self._polygons[label]) >= 3:
                    self._current_index += 1

            if key in (ord("s"), ord("S"), 19):
                if all(len(self._polygons[label]) >= 3 for label in self._labels):
                    return self._polygons


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return parsed


def _write_roi_config(
    output_path: Path,
    polygons: dict[str, list[tuple[int, int]]],
    base_config_path: Path,
    lane_count: int,
    step_colors: dict[str, tuple[int, int, int]],
) -> None:
    config = _read_yaml(base_config_path)

    config.setdefault("polygons", {})
    config["polygons"]["view"] = [list(point) for point in polygons["view"]]
    config["polygons"]["calibration"] = [list(point) for point in polygons["calibration"]]

    lane_keys = [f"lane-{index}" for index in range(1, lane_count + 1)]
    existing_lanes = config.get("lanes", [])
    if not isinstance(existing_lanes, list):
        existing_lanes = []

    lanes: list[dict[str, Any]] = []
    for index, lane_name in enumerate(lane_keys):
        existing = (
            existing_lanes[index]
            if index < len(existing_lanes) and isinstance(existing_lanes[index], dict)
            else {}
        )
        px_to_meter = existing.get("px_to_meter", 0.08)
        if not isinstance(px_to_meter, (int, float)) or float(px_to_meter) <= 0:
            px_to_meter = 0.08

        lane_color = step_colors.get(lane_name, (255, 255, 255))
        lanes.append(
            {
                "name": lane_name,
                "polygon": [list(point) for point in polygons[lane_name]],
                "px_to_meter": float(px_to_meter),
                "color": [int(channel) for channel in lane_color],
            }
        )

    config["lanes"] = lanes
    config.setdefault("overlay", {})
    # Disable hardcoded default lines for newly calibrated cameras.
    config["overlay"]["lines"] = []
    config["roi_steps"] = {
        label: {
            "color": [int(channel) for channel in step_colors[label]],
            "point_order": "clockwise_recommended",
        }
        for label in polygons.keys()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ROI polygons for a camera view.")
    parser.add_argument("--video", default="0", help="Video source path or camera index")
    parser.add_argument(
        "--lanes",
        type=int,
        default=None,
        help="Number of lane polygons to collect. Default: base config lane count or 5",
    )
    parser.add_argument(
        "--output",
        default="config/camera.yaml",
        help="YAML file to save polygon calibration",
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Base YAML to merge with before saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    base_config = _read_yaml(base_config_path)

    lane_count = args.lanes
    if lane_count is None:
        base_lanes = base_config.get("lanes")
        if isinstance(base_lanes, list) and len(base_lanes) > 0:
            lane_count = len(base_lanes)
        else:
            lane_count = DEFAULT_LANE_COUNT
    if lane_count < 1:
        raise ValueError("--lanes must be >= 1")

    labels = build_calibration_labels(lane_count)
    step_colors = build_step_colors(labels)

    print("Point order guidance: click points clockwise around each polygon.")
    print(
        "Mac shortcut tip: try single keys first (u/z/n/r/s). "
        "If not captured, use Cmd+u/z/n/r/s."
    )

    capture = open_video_source(args.video)
    if capture is None or not capture.isOpened():
        raise RuntimeError("Unable to open video source for ROI calibration.")

    ret, frame = capture.read()
    capture.release()
    if not ret or frame is None:
        raise RuntimeError("Unable to read a frame from the video source.")

    collector = PolygonCollector(frame=frame, labels=labels, step_colors=step_colors)

    try:
        polygons = collector.run()
    finally:
        cv.destroyAllWindows()

    if polygons is None:
        print("Calibration cancelled.")
        return

    output_path = Path(args.output)
    _write_roi_config(
        output_path=output_path,
        polygons=polygons,
        base_config_path=base_config_path,
        lane_count=lane_count,
        step_colors=step_colors,
    )
    print(f"Saved ROI calibration to {output_path}")


if __name__ == "__main__":
    main()
