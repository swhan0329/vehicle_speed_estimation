from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2 as cv

from app.io.video_source import open_video_source, resolve_fps
from app.speed.estimator import LaneSpeedEstimator
from app.speed.smoothing import SpeedSmoother
from app.track.lk_tracker import LKTracker
from app.types import RuntimeConfig
from app.viz.overlay import OverlayRenderer
from config.loader import load_config


def apply_px_to_meter_override(
    config: RuntimeConfig,
    px_to_meter_override: list[float] | None,
) -> RuntimeConfig:
    if px_to_meter_override is None:
        return config

    lane_count = len(config.lanes)
    if lane_count == 0:
        raise ValueError("No lanes available in config to apply px_to_meter override")

    if len(px_to_meter_override) == 1:
        values = px_to_meter_override * lane_count
    elif len(px_to_meter_override) == lane_count:
        values = px_to_meter_override
    else:
        raise ValueError(
            "px_to_meter override length mismatch: "
            f"expected 1 or {lane_count} values, got {len(px_to_meter_override)}"
        )

    updated_lanes = [
        replace(lane, px_to_meter=float(value))
        for lane, value in zip(config.lanes, values)
    ]
    return replace(config, lanes=updated_lanes)


def _create_writer(
    output_path: str | None,
    frame_size: tuple[int, int],
    fps: float,
) -> cv.VideoWriter | None:
    if not output_path:
        return None

    fourcc = cv.VideoWriter_fourcc(
        *("mp4v" if output_path.lower().endswith(".mp4") else "XVID")
    )
    writer = cv.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"Warning: unable to open output file: {output_path}")
        return None
    return writer


def run_pipeline(
    video_src: str,
    *,
    output_path: str | None,
    show: bool,
    config_path: str | Path | None,
    px_to_meter_override: list[float] | None = None,
) -> None:
    config = load_config(config_path=config_path)
    config = apply_px_to_meter_override(config, px_to_meter_override)

    capture = open_video_source(video_src)
    if capture is None or not capture.isOpened():
        raise RuntimeError("Unable to open video source.")

    ret, frame = capture.read()
    if not ret or frame is None:
        capture.release()
        raise RuntimeError("Unable to read first frame from source.")

    frame_h, frame_w = frame.shape[:2]
    fps = resolve_fps(capture, config.fallback_fps)
    writer = _create_writer(output_path, (frame_w, frame_h), fps)

    lk_params = {
        "winSize": config.lk_win_size,
        "maxLevel": config.lk_max_level,
        "criteria": (
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            config.lk_criteria_count,
            config.lk_criteria_eps,
        ),
    }
    feature_params = {
        "maxCorners": config.feature_max_corners,
        "qualityLevel": config.feature_quality_level,
        "minDistance": config.feature_min_distance,
        "blockSize": config.feature_block_size,
    }

    tracker = LKTracker(
        track_len=config.track_len,
        detect_interval=config.detect_interval,
        lk_params=lk_params,
        feature_params=feature_params,
    )
    estimator = LaneSpeedEstimator(lanes=config.lanes, fps=fps)
    smoother = SpeedSmoother(lane_count=len(config.lanes), smooth_coef=0.5)
    renderer = OverlayRenderer(config=config, frame_size=(frame_w, frame_h))

    frame_idx = 0

    try:
        while True:
            vis, cmask, frame_gray = renderer.prepare(frame)
            tracks = tracker.process_frame(frame_gray, frame_idx)

            speed_frame = estimator.estimate(tracks)
            lane_speeds = smoother.update(
                speed_frame.speeds_kmh,
                speed_frame.point_counts,
                update_enabled=frame_idx % config.detect_interval == 0,
                min_points_for_speed=config.min_points_for_speed,
            )

            renderer.draw(
                vis,
                cmask,
                tracks,
                lane_speeds,
                speed_frame.point_counts,
            )
            output_frame = renderer.blend(cmask, vis)

            if writer is not None:
                writer.write(output_frame)

            if show:
                cv.imshow(config.preview_window_name, output_frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            ret, frame = capture.read()
            frame_idx += 1
            if not ret or frame is None:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv.destroyAllWindows()
