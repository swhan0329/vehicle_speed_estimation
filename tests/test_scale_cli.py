import unittest

from app.calibrate.scale import ScaleError, resolve_pixel_distance


class TestScaleCli(unittest.TestCase):
    def test_resolve_pixel_distance_prefers_pixels(self):
        value = resolve_pixel_distance(
            pixels=123.4,
            point1=None,
            point2=None,
            interactive=False,
            video=None,
            pick_points_fn=lambda _video: ((0, 0), (10, 0)),
        )
        self.assertEqual(value, 123.4)

    def test_resolve_pixel_distance_uses_point_pair(self):
        value = resolve_pixel_distance(
            pixels=None,
            point1=(0, 0),
            point2=(3, 4),
            interactive=False,
            video=None,
            pick_points_fn=lambda _video: ((0, 0), (10, 0)),
        )
        self.assertEqual(value, 5.0)

    def test_resolve_pixel_distance_uses_interactive_video_default(self):
        calls = []

        def fake_pick(video: str):
            calls.append(video)
            return (0, 0), (6, 8)

        value = resolve_pixel_distance(
            pixels=None,
            point1=None,
            point2=None,
            interactive=True,
            video=None,
            pick_points_fn=fake_pick,
        )

        self.assertEqual(value, 10.0)
        self.assertEqual(calls, ["0"])

    def test_resolve_pixel_distance_raises_without_input(self):
        with self.assertRaises(ScaleError):
            resolve_pixel_distance(
                pixels=None,
                point1=None,
                point2=None,
                interactive=False,
                video=None,
                pick_points_fn=lambda _video: ((0, 0), (1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
