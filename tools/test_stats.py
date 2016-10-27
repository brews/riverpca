import unittest
from numpy import linspace, sin, tan, reshape, repeat
from numpy.testing import assert_almost_equal
from .tools import pearson_corr, composite_ttest

class TestStats(unittest.TestCase):

    def setUp(self):
        x = linspace(1, 15, 30)
        self.y1 = sin(x)
        self.y2_field = reshape(repeat(tan(x), 4), (30, 2, 2))

    def test_pearson_corr(self):
        # Maybe should break these down.
        test_r, test_p = pearson_corr(self.y1, self.y2_field)
        assert_almost_equal(test_r, 0.4056, decimal = 3)
        assert_almost_equal(test_p, 0.0261, decimal = 3)
        self.assertEqual(test_r.shape, (2, 2))
        self.assertEqual(test_p.shape, (2, 2))

    def test_composite_ttest(self):
        test_diff, test_p = composite_ttest(self.y1, self.y2_field)
        assert_almost_equal(test_diff, 2.2636, decimal = 3)
        assert_almost_equal(test_p, 0.1084, decimal = 3)
        self.assertEqual(test_diff.shape, (2, 2))
        self.assertEqual(test_p.shape, (2, 2))

