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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    run_pipeline(
        args.video_source,
        output_path=args.output,
        show=args.show,
        config_path=Path(args.config) if args.config else None,
    )
    print("Done")


if __name__ == "__main__":
    main()
