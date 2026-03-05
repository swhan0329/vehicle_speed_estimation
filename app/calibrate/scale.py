from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import cv2 as cv
import numpy as np
import yaml

from app.io.video_source import open_video_source


class ScaleError(ValueError):
    """Raised when scale calibration input is invalid."""


def compute_px_to_meter(pixel_distance: float, meter_distance: float) -> float:
    if pixel_distance <= 0:
        raise ScaleError("pixel_distance must be > 0")
    if meter_distance <= 0:
        raise ScaleError("meter_distance must be > 0")
    return meter_distance / pixel_distance


def _distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.subtract(point1, point2)))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ScaleError(f"Config file not found: {path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ScaleError("Config root must be a mapping")
    return parsed


def _write_yaml(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _pick_two_points(video_source: str) -> tuple[tuple[int, int], tuple[int, int]]:
    capture = open_video_source(video_source)
    if capture is None or not capture.isOpened():
        raise ScaleError("Unable to open video source for point picking.")

    ret, frame = capture.read()
    capture.release()
    if not ret or frame is None:
        raise ScaleError("Unable to read frame from video source.")

    picked: list[tuple[int, int]] = []
    window = "scale_calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv.EVENT_LBUTTONDOWN and len(picked) < 2:
            picked.append((x, y))

    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window, on_mouse)

    try:
        while True:
            canvas = frame.copy()
            message = "Click 2 points with known real-world distance"
            cv.putText(
                canvas,
                message,
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            for point in picked:
                cv.circle(canvas, point, 5, (0, 255, 255), -1)
            if len(picked) == 2:
                cv.line(canvas, picked[0], picked[1], (0, 255, 255), 2)
            cv.imshow(window, canvas)

            key = cv.waitKey(20) & 0xFF
            if key == ord("q"):
                raise ScaleError("Scale calibration cancelled by user.")
            if len(picked) == 2:
                break
    finally:
        cv.destroyWindow(window)

    return picked[0], picked[1]


def resolve_pixel_distance(
    *,
    pixels: float | None,
    point1: tuple[int, int] | None,
    point2: tuple[int, int] | None,
    interactive: bool,
    video: str | None,
    pick_points_fn: Callable[[str], tuple[tuple[int, int], tuple[int, int]]],
) -> float:
    if pixels is not None:
        return float(pixels)

    if point1 and point2:
        return _distance(point1, point2)

    if interactive or video:
        source = video if video else "0"
        selected_point1, selected_point2 = pick_points_fn(source)
        return _distance(selected_point1, selected_point2)

    raise ScaleError(
        "Provide --pixels, --point1/--point2, or --interactive/--click/--video "
        "for point picking."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate lane px_to_meter scale.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Input YAML config path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path. Defaults to --config in-place update",
    )
    parser.add_argument("--lane", type=int, required=True, help="1-based lane index")
    parser.add_argument("--meters", type=float, required=True, help="Real-world distance in meters")
    parser.add_argument(
        "--pixels",
        type=float,
        default=None,
        help="Pixel distance between two points. If omitted, use --point1/--point2 or --video",
    )
    parser.add_argument(
        "--point1",
        nargs=2,
        type=int,
        default=None,
        metavar=("X1", "Y1"),
        help="First pixel point",
    )
    parser.add_argument(
        "--point2",
        nargs=2,
        type=int,
        default=None,
        metavar=("X2", "Y2"),
        help="Second pixel point",
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Video source used by interactive point picking. "
            "Defaults to camera index 0 when --interactive/--click is used without --video."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive window and click two points to measure pixel distance.",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="Alias of --interactive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactive_enabled = bool(args.interactive or args.click)
    point1 = tuple(args.point1) if args.point1 else None
    point2 = tuple(args.point2) if args.point2 else None
    pixel_distance = resolve_pixel_distance(
        pixels=args.pixels,
        point1=point1,
        point2=point2,
        interactive=interactive_enabled,
        video=args.video,
        pick_points_fn=_pick_two_points,
    )

    px_to_meter = compute_px_to_meter(pixel_distance, args.meters)

    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else config_path
    config = _read_yaml(config_path)

    lanes = config.get("lanes")
    if not isinstance(lanes, list):
        raise ScaleError("Config must contain a lanes list")

    lane_index = args.lane - 1
    if lane_index < 0 or lane_index >= len(lanes):
        raise ScaleError(f"Lane index out of range: {args.lane}")
    if not isinstance(lanes[lane_index], dict):
        raise ScaleError(f"Lane entry must be a mapping: lanes[{lane_index}]")

    lanes[lane_index]["px_to_meter"] = float(px_to_meter)
    _write_yaml(output_path, config)

    print(
        f"Updated lane {args.lane} px_to_meter={px_to_meter:.6f} "
        f"(pixels={pixel_distance:.3f}, meters={args.meters:.3f})"
    )


if __name__ == "__main__":
    main()
