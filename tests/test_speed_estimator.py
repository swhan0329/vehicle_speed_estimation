import unittest

from app.speed.estimator import LaneSpeedEstimator
from app.types import LaneConfig


class TestLaneSpeedEstimator(unittest.TestCase):
    def test_estimate_computes_lane_speed_and_point_counts(self):
        lanes = [
            LaneConfig(
                name="lane-1",
                polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
                px_to_meter=0.1,
                color=(255, 0, 0),
            ),
            LaneConfig(
                name="lane-2",
                polygon=[(20, 0), (40, 0), (40, 20), (20, 20)],
                px_to_meter=0.2,
                color=(0, 255, 0),
            ),
        ]
        estimator = LaneSpeedEstimator(lanes=lanes, fps=10.0)

        tracks = [
            [(5.0, 5.0), (15.0, 5.0)],
            [(25.0, 5.0), (35.0, 5.0)],
        ]

        result = estimator.estimate(tracks)

        self.assertEqual(result.point_counts, [1, 1])
        self.assertAlmostEqual(result.speeds_kmh[0], 36.0, places=4)
        self.assertAlmostEqual(result.speeds_kmh[1], 72.0, places=4)

    def test_estimate_ignores_track_outside_lanes(self):
        lanes = [
            LaneConfig(
                name="lane-1",
                polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
                px_to_meter=0.1,
                color=(255, 0, 0),
            )
        ]
        estimator = LaneSpeedEstimator(lanes=lanes, fps=10.0)

        tracks = [[(30.0, 30.0), (35.0, 35.0)]]

        result = estimator.estimate(tracks)

        self.assertEqual(result.point_counts, [0])
        self.assertEqual(result.speeds_kmh, [0.0])


if __name__ == "__main__":
    unittest.main()
