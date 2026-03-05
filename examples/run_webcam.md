# Run Webcam

## Command
```bash
python main.py 0 --output outputs/webcam_output.mp4 --show
```

## Notes
- `0` is the default camera index.
- For a different camera, try `1` or `2`.
- Webcam perspective changes often, so lane speed can be noisy unless calibrated for that exact view.
