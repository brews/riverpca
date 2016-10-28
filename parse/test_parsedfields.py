"""Very basic unittests for data after it has been parsed
"""

import unittest
import os
import xarray as xr
from numpy.testing import assert_almost_equal
from pytest import mark


PARSED_TCRV2_PATH = '../data/tcrv2_z500_season.nc'
PARSED_ERSST_PATH = '../data/ersstv3b_season.nc'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@mark.skipif(not os.path.isfile(os.path.join(THIS_DIR, PARSED_TCRV2_PATH)),
             reason = 'parsed file does not exist')
class TestTcrv2(unittest.TestCase):

    def setUp(self):
        self.target_path = os.path.join(THIS_DIR, PARSED_TCRV2_PATH)
    
    def test_djf1982(self):
        # Maybe should break these down.
        with xr.open_dataset(self.target_path) as d:
            target = d.z500.sel(lat = 47.5, lon = 237.5,
                                time = '1982-12-01', method = 'nearest')
            assert_almost_equal(target.values, 5500.33, decimal = 1)

