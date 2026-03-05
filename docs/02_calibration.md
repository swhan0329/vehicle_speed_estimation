# Calibration Guide

This project requires per-camera calibration.

## Most common setup mistakes (read first)
- Point order: click polygon points clockwise for `view`, `calibration`, and all lanes.
- Resolution mismatch: if input video resolution changes, recalibrate ROI and lane scale.
- Missing scale: every lane needs `px_to_meter` (`meters / pixels`) and values differ by lane/camera.

## ROI calibration
Collect these polygons in order:
1. `view`: visible output region
2. `calibration`: region where optical-flow points are measured
3. `lane-1` to `lane-N`: lane assignment polygons (`--lanes N`)

Command:
```bash
python -m app.calibrate.roi --video path/to/video.mp4 --lanes 5 --output config/camera.yaml
```

Keys:
- Left click: add point
- `n` / `Enter`: next polygon (requires at least 3 points)
- `u` / `z`: undo last point
- `r`: reset current polygon
- `s`: save
- `q` / `ESC`: cancel

Point direction recommendation:
- Click points clockwise for every polygon (`view`, `calibration`, and each lane).

Mac note:
- Try single-key shortcuts first.
- On some macOS environments, key input is only detected with `Cmd` pressed.
- If single keys do not work, use `Cmd+u/z/n/r/s`.

## Scale calibration (`px_to_meter`)
Each lane needs `px_to_meter = meters / pixels`.

`px_to_meter` changes with road geometry and camera perspective.
You should recalibrate per camera and per road.

Recommended real-world references:
- lane width in the target lane
- stop-line or road marking interval
- known vehicle body length/width (as an approximate reference)

Options:
- Direct pixel distance:
```bash
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --pixels 142
```
- Two points:
```bash
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --point1 100 220 --point2 240 220
```
- Interactive point picking:
```bash
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --interactive --video path/to/video.mp4
```
- Interactive alias:
```bash
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --click --video path/to/video.mp4
```

## Validation error examples
If YAML shape is invalid, config loader raises explicit messages:
- `polygons.view must contain at least 3 points`
- `lanes must be a non-empty list`
- `lanes[0].px_to_meter must be > 0`

## Run with calibrated config
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --output output.mp4 --show
```

Optional runtime override:
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --px-to-meter 0.0895,0.088,0.0774 --output output.mp4 --show
```

If a single value is provided, it is applied to all lanes:
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --px-to-meter 0.082 --output output.mp4 --show
```

## Overlay rendering check
After calibration, the rendered output should show:
- Orange boundary for `view`
- Yellow boundary for `calibration`
- Per-lane colored filled polygons (`lane-1 ... lane-N`)

If the rendered shape still looks wrong:
- Verify you used the same video for ROI collection and final run.
- Verify you passed `--config config/camera.yaml` when running `main.py`.
- Re-run ROI calibration if you changed monitor scale/window size settings.
