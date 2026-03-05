# Calibrate a New Camera

## 1) Draw ROI polygons
```bash
python -m app.calibrate.roi --video path/to/video.mp4 --lanes 3 --output config/camera.yaml
```
Expected order:
1. `view`
2. `calibration`
3. `lane-1` ... `lane-N`

## 2) Set lane scale (`px_to_meter`)
Use measured distance from your real scene (for example lane width `3.5m`).

Interactive picking:
```bash
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --interactive --video path/to/video.mp4
python -m app.calibrate.scale --config config/camera.yaml --lane 2 --meters 3.5 --interactive --video path/to/video.mp4
python -m app.calibrate.scale --config config/camera.yaml --lane 3 --meters 3.5 --interactive --video path/to/video.mp4
```

## 3) Run with calibrated config
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --output outputs/calibrated_output.mp4 --show
```

## Calibration reminder
- Recalibrate if camera angle, zoom, or input resolution changes.
- `px_to_meter` differs by lane due to perspective.
