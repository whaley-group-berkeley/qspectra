import numpy as np
import unittest
from numpy import pi
from numpy.testing import assert_equal, assert_allclose

from qspectra import polarization


class TestPolarization(unittest.TestCase):
    def test_polarization_vector(self):
        assert_equal(polarization.polarization_vector('x'),
                     [1, 0, 0])
        assert_equal(polarization.polarization_vector(0),
                     [1, 0, 0])
        assert_allclose(polarization.polarization_vector(pi / 2),
                        [0, 1, 0], atol=1e-10)
        with self.assertRaises(ValueError):
            polarization.polarization_vector('X')
        with self.assertRaises(ValueError):
            polarization.polarization_vector([1, 0])

    def test_invariant_weights_4th_order(self):
        # Hochstrasser, Eq. 9(a):
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        np.ones(3) / 15)
        # Hochstrasser, Eq. 9(b):
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        np.array([4, -1, -1]) / 30.0)
        # magic angle:
        ma = polarization.MAGIC_ANGLE
        assert_allclose(
            polarization.invariant_weights_4th_order([0, 0, ma, ma]),
            [1 / 9.0, 0, 0], atol=1e-10)
        # see:
        # Zanni, M. T., Ge, N. H., Kim, Y. S. & Hochstrasser, R. M. Two-
        # dimensional IR spectroscopy can be designed to eliminate the diagonal
        # peaks and expose only the crosspeaks needed for structure
        # determination. Proc. Natl Acad. Sci. USA 98, 11265-11270 (2001).
        assert_allclose(
            polarization.invariant_weights_4th_order(
                [pi / 4, -pi / 4, pi / 2, 0]),
            [0, 1 / 12.0, -1 / 12.0], atol=1e-10)
        # consistency checks:
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        polarization.invariant_weights_4th_order('yyyy'))
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        polarization.invariant_weights_4th_order('zzzz'))
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        polarization.invariant_weights_4th_order('yyxx'))
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        polarization.invariant_weights_4th_order('zzyy'))

    def test_invariant_polarizations(self):
        with self.assertRaises(ValueError):
            polarization.invariant_polarizations(((0, 1), (0, 2)))
        invariants = polarization.FOURTH_ORDER_INVARIANTS
        for invariant in invariants:
            self.assertEqual(len(polarization.invariant_polarizations(invariant)),
                             9)
        self.assertItemsEqual(polarization.invariant_polarizations(invariants[0]),
                              ['xxxx', 'xxyy', 'xxzz',
                               'yyxx', 'yyyy', 'yyzz',
                               'zzxx', 'zzyy', 'zzzz'])

    def test_random_rotation_matrix(self):
        M = polarization.random_rotation_matrix()
        self.assertTrue(isinstance(M, np.ndarray))
        self.assertEqual(M.shape, (3, 3))
