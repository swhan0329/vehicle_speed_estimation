import unittest

import numpy as np

from app.viz.overlay import rasterize_polygon_mask


class TestOverlayMask(unittest.TestCase):
    def test_rasterize_polygon_mask_preserves_concave_shape(self):
        polygon = [(2, 1), (8, 1), (8, 4), (5, 4), (5, 8), (2, 8)]

        mask = rasterize_polygon_mask(frame_size=(12, 12), polygon=polygon)

        self.assertEqual(mask[7, 7], 0)
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(mask[5, 3], 1)


if __name__ == "__main__":
    unittest.main()
