# Run Video

## Command
```bash
python main.py path/to/video.mp4 --output outputs/video_output.mp4 --show
```

## With custom calibration config
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --output outputs/video_output.mp4 --show
```

## Optional lane scale override
```bash
python main.py path/to/video.mp4 --config config/camera.yaml --px-to-meter 0.0895,0.088,0.0774 --output outputs/video_output.mp4 --show
```
