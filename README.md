## Vehicle Speed Estimation

### Overview
This project uses the optical flow algorithm, specifically the Lucas-Kanade tracker, to estimate vehicle speeds from mono camera (CCTV) footage.

### Prerequisites
- Python 3.x
- Required libraries: `opencv-python`, `numpy`

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/swhan0329/vehicle_speed_estimation.git
   cd vehicle_speed_estimation
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **With an Input Video**
   ```bash
   python main.py [input video name]
   ```
   Replace `[input video name]` with the path to your video file.

2. **Without an Input Video**
   The script will automatically use the webcam on your computer.
   ```bash
   python main.py
   ```

### File Descriptions

- **main.py**: The main script to run the vehicle speed estimation.
- **video.py**: Contains functions to handle video input and processing.
- **common.py**: Includes common functions and utilities used across the project.
- **tst_scene_render.py**: Test and render scenes for visualization.

### Example
To run the speed estimation on a sample video:
```bash
python main.py sample_video.mp4
```

To use the webcam for live speed estimation:
```bash
python main.py
```

### Additional Notes
- Ensure that your video has a clear view of the road and vehicles for accurate speed estimation.
- Adjust parameters in `common.py` if needed to fit specific requirements or to improve performance.

### Running Tests
Execute the unit tests with Python's built-in test runner:
```bash
python -m unittest discover -s tests
```
