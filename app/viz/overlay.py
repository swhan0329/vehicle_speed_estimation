from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from app.types import RuntimeConfig, Track
from common import draw_str


@dataclass(frozen=True)
class TextLayout:
    left_x: int
    right_margin: int
    row_ys: list[int]
    font_scale: float
    thickness: int


def compute_text_layout(
    frame_size: tuple[int, int],
    lane_count: int,
) -> TextLayout:
    frame_w, frame_h = frame_size
    if lane_count <= 0:
        return TextLayout(
            left_x=max(12, int(frame_w * 0.025)),
            right_margin=max(12, int(frame_w * 0.025)),
            row_ys=[],
            font_scale=max(0.55, min(1.25, frame_w / 1280.0)),
            thickness=2 if frame_w < 1000 else 3,
        )

    top_margin = max(24, int(frame_h * 0.055))
    bottom_margin = max(16, int(frame_h * 0.04))
    default_spacing = max(18, int(frame_h * 0.055))

    if lane_count == 1:
        row_ys = [top_margin]
    else:
        max_spacing = max(1, (frame_h - top_margin - bottom_margin) // (lane_count - 1))
        row_spacing = min(default_spacing, max_spacing)
        row_ys = [top_margin + index * row_spacing for index in range(lane_count)]

    return TextLayout(
        left_x=max(12, int(frame_w * 0.025)),
        right_margin=max(12, int(frame_w * 0.025)),
        row_ys=row_ys,
        font_scale=max(0.55, min(1.25, frame_w / 1280.0)),
        thickness=2 if frame_w < 1000 else 3,
    )


def rasterize_polygon_mask(
    frame_size: tuple[int, int],
    polygon: list[tuple[int, int]],
) -> np.ndarray:
    frame_w, frame_h = frame_size
    mask = np.zeros((frame_h, frame_w), np.uint8)
    polygon_array = np.asarray(polygon, dtype=np.int32)
    cv.fillPoly(mask, [polygon_array], 1)
    return mask


class OverlayRenderer:
    """Render masks, lane overlays, and tracking diagnostics."""

    def __init__(self, config: RuntimeConfig, frame_size: tuple[int, int]) -> None:
        frame_w, frame_h = frame_size
        self._config = config

        self._view_polygon = np.asarray(config.view_polygon, dtype=np.int32)
        self._calibration_polygon = np.asarray(config.calibration_polygon, dtype=np.int32)
        self._lane_polygons = [np.asarray(lane.polygon, dtype=np.int32) for lane in config.lanes]

        self._calibration_mask = rasterize_polygon_mask(
            frame_size=(frame_w, frame_h),
            polygon=config.calibration_polygon,
        )
        self._view_mask = rasterize_polygon_mask(
            frame_size=(frame_w, frame_h),
            polygon=config.view_polygon,
        )

    def prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vis = cv.bitwise_and(frame, frame, mask=self._view_mask)
        cmask = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.bitwise_and(frame_gray, frame_gray, mask=self._calibration_mask)
        return vis, cmask, frame_gray

    def draw(
        self,
        vis: np.ndarray,
        cmask: np.ndarray,
        tracks: list[Track],
        lane_speeds: list[float],
        lane_point_counts: list[int],
    ) -> None:
        frame_h, frame_w = vis.shape[:2]
        text_layout = compute_text_layout(
            frame_size=(frame_w, frame_h),
            lane_count=len(self._config.lanes),
        )

        # Draw polygon outlines to make calibration/view boundaries explicit.
        cv.polylines(vis, [self._view_polygon], True, (0, 180, 255), 2)
        cv.polylines(vis, [self._calibration_polygon], True, (0, 255, 255), 2)
        label_y = max(18, text_layout.row_ys[0] - 14) if text_layout.row_ys else 18
        label_scale = max(0.45, text_layout.font_scale * 0.75)
        draw_str(
            vis,
            (text_layout.left_x, label_y),
            "view",
            color=(0, 180, 255),
            thickness=2,
            font_scale=label_scale,
        )
        draw_str(
            vis,
            (text_layout.left_x + 90, label_y),
            "calibration",
            color=(0, 255, 255),
            thickness=2,
            font_scale=label_scale,
        )

        for line in self._config.overlay_lines:
            cv.line(vis, line.start, line.end, line.color, line.thickness)

        for lane, polygon in zip(self._config.lanes, self._lane_polygons):
            cv.fillPoly(cmask, [polygon], lane.color, cv.LINE_AA)

        for lane_index, lane in enumerate(self._config.lanes):
            speed_text = f"{lane_index + 1}-lane speed: {int(lane_speeds[lane_index])} km/h"
            count_text = f"ptn{lane_index + 1}: {lane_point_counts[lane_index]}"
            y_pos = text_layout.row_ys[lane_index]

            draw_str(
                vis,
                (text_layout.left_x, y_pos),
                speed_text,
                thickness=text_layout.thickness,
                font_scale=text_layout.font_scale,
            )

            text_size, _baseline = cv.getTextSize(
                count_text,
                cv.FONT_HERSHEY_SIMPLEX,
                text_layout.font_scale,
                text_layout.thickness,
            )
            right_x = max(
                text_layout.left_x + 8,
                frame_w - text_layout.right_margin - text_size[0],
            )
            draw_str(
                vis,
                (right_x, y_pos),
                count_text,
                thickness=text_layout.thickness,
                font_scale=text_layout.font_scale,
            )

        for track in tracks:
            if not track:
                continue
            x, y = track[-1]
            cv.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

        if tracks:
            polylines = [np.int32(track) for track in tracks if len(track) >= 2]
            if polylines:
                cv.polylines(vis, polylines, False, (0, 0, 255))

    def blend(self, cmask: np.ndarray, vis: np.ndarray) -> np.ndarray:
        blended = vis.copy()
        cv.addWeighted(cmask, self._config.alpha, vis, 1 - self._config.alpha, 0, blended)
        return blended
