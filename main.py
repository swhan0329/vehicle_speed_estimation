#!/usr/bin/env python

"""Vehicle speed estimation using the Lucas-Kanade tracker.

Usage:
    python main.py [video_source] [--output OUTPUT] [--show] [--config CONFIG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.pipeline import run_pipeline


def parse_px_to_meter_values(raw: str) -> list[float]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("px_to_meter list is empty")

    values: list[float] = []
    for token in tokens:
        value = float(token)
        if value <= 0:
            raise ValueError("px_to_meter must be > 0")
        values.append(value)
    return values


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle speed estimation from video input."
    )
    parser.add_argument(
        "video_source",
        nargs="?",
        default="0",
        help="Video source path or camera index.",
    )
    parser.add_argument("--output", default="output.mp4", help="Output video path.")
    parser.add_argument("--show", action="store_true", help="Show preview window.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path. Falls back to config/default.yaml.",
    )
    parser.add_argument(
        "--px-to-meter",
        default=None,
        help=(
            "Optional override for lane scale values (meters per pixel). "
            "Use one value for all lanes or comma-separated values per lane. "
            "Examples: 0.082 or 0.0895,0.088,0.0774"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    px_to_meter_override = (
        parse_px_to_meter_values(args.px_to_meter)
        if args.px_to_meter is not None
        else None
    )
    run_pipeline(
        args.video_source,
        output_path=args.output,
        show=args.show,
        config_path=Path(args.config) if args.config else None,
        px_to_meter_override=px_to_meter_override,
    )
    print("Done")


if __name__ == "__main__":
    main()
