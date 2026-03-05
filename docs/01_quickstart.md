# Quickstart

## 1) Install
```bash
git clone https://github.com/swhan0329/vehicle_speed_estimation.git
cd vehicle_speed_estimation
pip install -r requirements.txt
```

## 2) Run with default config
```bash
python main.py path/to/video.mp4 --output output.mp4 --show
```

## 3) Run with a custom camera config
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --output output.mp4 --show
```

Optional: quick lane scale override at runtime
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --px-to-meter 0.0895,0.088,0.0774 --output output.mp4 --show
```

## 4) Minimal commands for calibration
```bash
python -m app.calibrate.roi --video path/to/video.mp4 --lanes 5 --output config/camera.yaml
python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --point1 100 220 --point2 240 220
```
