import unittest

import numpy as np
from numpy.testing import assert_allclose

from qspectra import bath


class TestDebyeBath(unittest.TestCase):
    def test_corr_func_consistency(self):
        test_bath = bath.DebyeBath(30, 50, 100)
        points = np.linspace(-200, 200, num=100)
        assert_allclose(np.real(list(map(test_bath.corr_func_complex, points))),
                        list(map(test_bath.corr_func_real, points)), atol=0.1)
