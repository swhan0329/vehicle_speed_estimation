import unittest
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without numpy
    NUMPY_AVAILABLE = False
    np = None

if NUMPY_AVAILABLE:
    import common

@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is required")
class TestCommonFunctions(unittest.TestCase):
    def test_anorm(self):
        vec = np.array([3.0, 4.0])
        self.assertAlmostEqual(common.anorm(vec), 5.0)
        self.assertAlmostEqual(common.anorm2(vec), 25.0)

    def test_mdot(self):
        a = np.eye(2)
        b = np.array([[2, 0], [0, 2]])
        result = common.mdot(a, b)
        np.testing.assert_array_equal(result, b)

    def test_rect2rect_mtx(self):
        src = (0, 0, 1, 1)
        dst = (0, 0, 2, 2)
        M = common.rect2rect_mtx(src, dst)
        expected = np.array([[2.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(M, expected)

if __name__ == '__main__':
    unittest.main()
