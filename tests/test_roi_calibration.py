import tempfile
import unittest
from pathlib import Path

import yaml

from app.calibrate.roi import (
    _write_roi_config,
    build_calibration_labels,
    build_step_colors,
    map_display_point_to_frame,
)


class TestRoiCalibrationHelpers(unittest.TestCase):
    def test_map_display_point_to_frame_scaled_window(self):
        mapped = map_display_point_to_frame(
            point=(320, 180),
            display_size=(640, 360),
            frame_size=(1280, 720),
        )

        self.assertTrue(639 <= mapped[0] <= 641)
        self.assertTrue(359 <= mapped[1] <= 361)

    def test_map_display_point_to_frame_clamps_out_of_bounds(self):
        mapped = map_display_point_to_frame(
            point=(9999, -100),
            display_size=(640, 360),
            frame_size=(1280, 720),
        )

        self.assertEqual(mapped, (1279, 0))

    def test_build_calibration_labels_with_custom_lane_count(self):
        labels = build_calibration_labels(3)
        self.assertEqual(
            labels,
            ["view", "calibration", "lane-1", "lane-2", "lane-3"],
        )

    def test_build_step_colors_has_color_for_every_label(self):
        labels = build_calibration_labels(6)
        colors = build_step_colors(labels)

        self.assertEqual(set(colors.keys()), set(labels))
        self.assertEqual(len(colors["view"]), 3)
        self.assertEqual(len(colors["lane-6"]), 3)

    def test_write_roi_config_respects_lane_count_and_step_colors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base.yaml"
            out_path = Path(temp_dir) / "camera.yaml"

            base_content = {
                "lanes": [
                    {
                        "name": "lane-1",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                        "px_to_meter": 0.1,
                        "color": [9, 9, 9],
                    },
                    {
                        "name": "lane-2",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                        "px_to_meter": 0.2,
                        "color": [8, 8, 8],
                    },
                ]
            }
            base_path.write_text(yaml.safe_dump(base_content), encoding="utf-8")

            labels = build_calibration_labels(3)
            colors = build_step_colors(labels)
            polygons = {
                "view": [[10, 10], [200, 10], [200, 120]],
                "calibration": [[20, 20], [160, 20], [160, 100]],
                "lane-1": [[30, 30], [70, 30], [70, 100]],
                "lane-2": [[80, 30], [120, 30], [120, 100]],
                "lane-3": [[130, 30], [170, 30], [170, 100]],
            }

            _write_roi_config(
                output_path=out_path,
                polygons=polygons,
                base_config_path=base_path,
                lane_count=3,
                step_colors=colors,
            )

            saved = yaml.safe_load(out_path.read_text(encoding="utf-8"))

            self.assertEqual(len(saved["lanes"]), 3)
            self.assertEqual(saved["lanes"][0]["px_to_meter"], 0.1)
            self.assertEqual(saved["lanes"][1]["px_to_meter"], 0.2)
            self.assertEqual(saved["lanes"][2]["color"], list(colors["lane-3"]))
            self.assertEqual(saved["roi_steps"]["view"]["color"], list(colors["view"]))
            self.assertEqual(saved["overlay"]["lines"], [])


if __name__ == "__main__":
    unittest.main()
