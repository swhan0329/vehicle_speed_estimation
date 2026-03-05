# Calibration Guide

This project requires per-camera calibration.

## ROI calibration
Collect these polygons in order:
1. `view`: visible output region
2. `calibration`: region where optical-flow points are measured
3. `lane-1` to `lane-N`: lane assignment polygons (`--lanes N`)

Command:
```bash
python -m app.calibrate.roi --video sample_video.mp4 --lanes 5 --output config/camera.yaml
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
- In OpenCV windows, modifier shortcuts (for example `Cmd+Z`) are not always captured consistently.
- Use single-key shortcuts above for stable behavior.

## Scale calibration (`px_to_meter`)
Each lane needs `px_to_meter = meters / pixels`.

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
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --video sample_video.mp4
```

## Run with calibrated config
```bash
python main.py sample_video.mp4 --config config/camera.yaml --output output.mp4 --show
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
