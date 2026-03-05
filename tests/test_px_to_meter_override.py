import unittest

from app.pipeline import apply_px_to_meter_override
from app.types import LaneConfig, OverlayLine, RuntimeConfig
from main import parse_px_to_meter_values


def make_runtime_config() -> RuntimeConfig:
    lanes = [
        LaneConfig(name="lane-1", polygon=[(0, 0), (1, 0), (1, 1)], px_to_meter=0.1, color=(1, 2, 3)),
        LaneConfig(name="lane-2", polygon=[(0, 0), (1, 0), (1, 1)], px_to_meter=0.2, color=(4, 5, 6)),
        LaneConfig(name="lane-3", polygon=[(0, 0), (1, 0), (1, 1)], px_to_meter=0.3, color=(7, 8, 9)),
    ]
    return RuntimeConfig(
        view_polygon=[(0, 0), (100, 0), (100, 100)],
        calibration_polygon=[(10, 10), (90, 10), (90, 90)],
        lanes=lanes,
        overlay_lines=[OverlayLine(start=(0, 0), end=(1, 1))],
    )


class TestPxToMeterOverride(unittest.TestCase):
    def test_parse_px_to_meter_values_parses_csv(self):
        values = parse_px_to_meter_values("0.09, 0.08,0.07")
        self.assertEqual(values, [0.09, 0.08, 0.07])

    def test_parse_px_to_meter_values_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            parse_px_to_meter_values("0.09,0")

    def test_apply_px_to_meter_override_single_value_applies_all_lanes(self):
        config = make_runtime_config()
        updated = apply_px_to_meter_override(config, [0.05])

        self.assertEqual([lane.px_to_meter for lane in updated.lanes], [0.05, 0.05, 0.05])

    def test_apply_px_to_meter_override_per_lane_values(self):
        config = make_runtime_config()
        updated = apply_px_to_meter_override(config, [0.11, 0.22, 0.33])

        self.assertEqual([lane.px_to_meter for lane in updated.lanes], [0.11, 0.22, 0.33])

    def test_apply_px_to_meter_override_rejects_length_mismatch(self):
        config = make_runtime_config()
        with self.assertRaises(ValueError):
            apply_px_to_meter_override(config, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
