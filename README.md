## Vehicle Speed Estimation

### Overview
This project estimates vehicle speed from fixed monocular CCTV footage using the Lucas-Kanade optical flow tracker.
It is designed as a beginner-friendly reference implementation that can be reused across cameras through ROI and scale calibration.

### Applications
- Traffic monitoring
- ITS (Intelligent Transportation Systems) research
- Computer vision education

### Demo
[![Vehicle speed estimation demo](assets/demo.gif)](https://www.youtube.com/shorts/AEd7tev39Ns)

Click the GIF to watch the full YouTube Shorts demo.

### Calibration Snapshot
1. `view`
![Calibration step view](assets/스크린샷 2026-03-05 오전 9.42.03.png)

2. `calibration`
![Calibration step calibration](assets/스크린샷 2026-03-05 오전 9.42.16.png)

3. `lane`
![Calibration step lane](assets/스크린샷 2026-03-05 오전 9.42.40.png)

### Quickstart
1. Clone and install dependencies.
   ```bash
   git clone https://github.com/swhan0329/vehicle_speed_estimation.git
   cd vehicle_speed_estimation
   pip install -r requirements.txt
   ```
2. Run with default calibration.
   ```bash
   python main.py sample_video.mp4 --output output.mp4 --show
   ```
3. Use a custom config for your camera.
   ```bash
   python main.py sample_video.mp4 --config config/camera.yaml --output output.mp4 --show
   ```

### Calibration Workflow
1. Collect polygons (view, calibration area, lane polygons).
   ```bash
   python -m app.calibrate.roi --video sample_video.mp4 --lanes 5 --output config/camera.yaml
   ```
2. Set lane scale (`px_to_meter`) with known real-world distance.
   ```bash
   python -m app.calibrate.scale --config config/camera.yaml --lane 1 --meters 3.5 --point1 100 220 --point2 240 220
   ```
3. Run pipeline with calibrated config.
   ```bash
   python main.py sample_video.mp4 --config config/camera.yaml --output output.mp4 --show
   ```

Point order recommendation: click points clockwise for every polygon.  
Mac shortcut note: OpenCV windows are most reliable with single keys (`u/z/n/r/s`, `Enter`, `ESC`).

### `px_to_meter` Setup (Important)
`px_to_meter` is camera-dependent and road-dependent.

- It must be recalibrated for each road and camera angle.
- Prefer real-world references measured at the target lane position:
  - lane width
  - stop-line spacing
  - known vehicle body length/width

Recommended workflow:
1. Use `app.calibrate.roi` to define lane polygons.
2. Use `app.calibrate.scale` with measured distance to update each lane scale.
3. Validate speeds on a short clip and fine-tune per lane if needed.

Optional runtime override (without editing YAML):
```bash
python main.py sample_video.mp4 --config config/camera.yaml --px-to-meter 0.0895,0.088,0.0774
```

If one value is provided, it is applied to all lanes:
```bash
python main.py sample_video.mp4 --config config/camera.yaml --px-to-meter 0.082
```

### Project Structure
- `main.py`: CLI entrypoint.
- `app/pipeline.py`: frame loop orchestration.
- `app/io/`: video source handling.
- `app/detect/`: feature detection module.
- `app/track/`: Lucas-Kanade tracker state/update.
- `app/speed/`: lane assignment, speed estimation, smoothing.
- `app/viz/`: overlays and debug rendering.
- `app/calibrate/`: ROI/scale calibration CLIs.
- `config/default.yaml`: default polygons, lane scales, runtime params.
- `config/loader.py`: YAML load/validation/merge.
- `tests/`: unit tests.

### Docs
- [Quickstart](docs/01_quickstart.md)
- [Calibration Guide](docs/02_calibration.md)
- [Common Failure Modes](docs/03_common_failure_modes.md)

### Notes
- Works best with fixed camera road scenes.
- You must recalibrate ROI polygons and lane scale values per camera view.
- Performance and accuracy can degrade with severe camera shake, heavy occlusion, or low frame rate footage.

### Running Tests
```bash
python -m unittest discover -s tests
```

### Contributing
Contributions are welcome.

- For bug fixes, docs, calibration UX, and performance improvements, please open a PR.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

### License
This project is licensed under the Apache License 2.0.  
See [LICENSE](LICENSE) for details.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=swhan0329/vehicle_speed_estimation&type=Date)](https://star-history.com/#swhan0329/vehicle_speed_estimation&Date)
