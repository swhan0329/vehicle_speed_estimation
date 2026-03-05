import tempfile
import unittest
from pathlib import Path

import yaml

from config.loader import ConfigError, load_config


class TestConfigLoader(unittest.TestCase):
    def test_load_config_with_default_has_expected_lanes(self):
        config = load_config()
        self.assertEqual(len(config.lanes), 5)
        self.assertEqual(config.detect_interval, 2)

    def test_load_config_with_override_updates_runtime_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override_path = Path(temp_dir) / "override.yaml"
            override_content = {
                "runtime": {
                    "detect_interval": 4,
                    "min_points_for_speed": 3,
                }
            }
            override_path.write_text(yaml.safe_dump(override_content), encoding="utf-8")

            config = load_config(config_path=override_path)

            self.assertEqual(config.detect_interval, 4)
            self.assertEqual(config.min_points_for_speed, 3)

    def test_load_config_with_invalid_lane_scale_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid.yaml"
            invalid_content = {
                "lanes": [
                    {
                        "name": "lane-1",
                        "polygon": [[0, 0], [1, 0], [1, 1]],
                        "px_to_meter": -1.0,
                        "color": [255, 0, 0],
                    }
                ]
            }
            invalid_path.write_text(yaml.safe_dump(invalid_content), encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(config_path=invalid_path)


if __name__ == "__main__":
    unittest.main()
