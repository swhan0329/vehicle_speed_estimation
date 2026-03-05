import unittest

from app.viz.overlay import compute_text_layout


class TestOverlayTextLayout(unittest.TestCase):
    def test_layout_scales_with_resolution(self):
        small = compute_text_layout(frame_size=(640, 360), lane_count=5)
        large = compute_text_layout(frame_size=(1920, 1080), lane_count=5)

        self.assertGreater(large.left_x, small.left_x)
        self.assertGreater(large.row_ys[0], small.row_ys[0])
        self.assertGreater(large.font_scale, small.font_scale)

    def test_layout_rows_stay_inside_frame(self):
        layout = compute_text_layout(frame_size=(640, 360), lane_count=5)

        self.assertEqual(len(layout.row_ys), 5)
        self.assertTrue(all(y > 0 for y in layout.row_ys))
        self.assertTrue(all(b > a for a, b in zip(layout.row_ys, layout.row_ys[1:])))
        self.assertLess(layout.row_ys[-1], 360)


if __name__ == "__main__":
    unittest.main()
